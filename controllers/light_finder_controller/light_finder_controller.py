"""light_finder_controller controller."""

from controller import Supervisor
from robot_network import RobotNetwork
from genetic_algorithm import GeneticAlgorithm
from utils import read_json_to_dict
import torch
import enum
import json
import uuid
import numpy as np

def load_robot_weights(robot_network, weights):
    idx = 0
    for param in robot_network.parameters():
        layer_size = param.data.numel()
        layer_weights = weights[idx:idx + layer_size]
        param.data = torch.tensor(layer_weights).reshape(param.data.shape)
        idx += layer_size


def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values


def normalize_sensor_values(sensor_values, min_value, max_value):
    normalized = [(x - min_value) / (max_value - min_value) for x in sensor_values]
    return normalized


def normalize_rotation_angle(rotation_angle):
    if rotation_angle < 0:
        rotation_angle += 2 * np.pi
    return rotation_angle


def get_sensor_angles(start_angle, degrees):
    radians = np.radians(degrees)
    angles = start_angle + radians
    angles = ((angles + np.pi) % (2 * np.pi)) - np.pi
    return angles


def save_data(initial_position, initial_rotation, light_position, ga_history):
    # Create a dictionary with world information and individuals data
    data_to_save = {
        "world_info": {
            "initial_position": initial_position,
            "initial_rotation": initial_rotation,
            "light_position": light_position
        },
        "individuals": []
    }
    
    # Add each individual's data
    for generation, individual in enumerate(ga_history):
        individual_data = {
            "generation": generation,
            "fitness": individual.fitness,
            "weights": individual.weights
        }
        data_to_save["individuals"].append(individual_data)
    
    filename = f"ga_history_{uuid.uuid4()}.json"
    with open(filename, "w") as f:
        json.dump(data_to_save, f)


class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    EXECUTION = 2   # Mode for using the trained controller


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
MAX_TIME = 30

robot_node = robot.getFromDef("MAIN1")
light_node = robot.getFromDef("LIGHT")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

light_translation_field = light_node.getField("translation")

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()

light_position = light_translation_field.getSFVec3f()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

degree_sensors = [20, 45, 90, 150, 210, 270, 315, 340]

light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]
light_sensors = []

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

mode = Mode.EXECUTION

if mode == Mode.TRAINING:

    POPULATION_SIZE = 100
    GENERATIONS = 25

    current_individual = 0
    current_generation = 0
    current_time = 0
    previous_time = 0

    robot_network = RobotNetwork()

    ga_history = []
    genetic_algorithm = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.05,
        representation="binary"
    )

    population = genetic_algorithm.generate_initial_population()

    weights_network = population[0].get_weights()
    load_robot_weights(robot_network, weights_network)

    light_history = []

    def fitness(light_history):
        light_data = np.array(light_history)
        complement = 1 - light_data
        fitness = np.sum(complement)
        return fitness

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo
            population[current_individual].fitness = fitness(light_history)
            #print(f"Generation: {current_generation}, Individual: {current_individual}, Fitness: {population[current_individual].fitness}")

            current_individual += 1

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            robot_node.resetPhysics()

            translation_field.setSFVec3f(INITIAL_POSITION)
            rotation_field.setSFRotation(INITIAL_ROTATION)

            if current_individual < POPULATION_SIZE: # cargamos pesos del individuo
                weights_network = population[current_individual].get_weights()
                load_robot_weights(robot_network, weights_network)

            light_history = []

            robot.step(timestep)

            if current_individual == POPULATION_SIZE:
                current_individual = 0
                fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")

                ga_history.append(fittest_individual)
                population = new_population
                current_generation += 1

                weights_network = population[0].get_weights()
                load_robot_weights(robot_network, weights_network)

                robot.step(timestep)
                if current_generation == GENERATIONS:
                    save_data(INITIAL_POSITION, INITIAL_ROTATION, light_position, ga_history)

        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4096)
        
        light_history.append(normalized_light_sensor_values)

        normalized_sensor_values = torch.tensor(normalized_light_sensor_values)
        directions = robot_network.forward(normalized_sensor_values)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)

elif mode == Mode.EXECUTION:
    file_path = "ga_history_20099ce8-c16c-4445-8949-d4e09cbd9028.json"
    data_dict = read_json_to_dict(file_path)

    robot_network = RobotNetwork()
    weights_network = data_dict["individuals"][-1]["weights"]
    load_robot_weights(robot_network, weights_network)

    translation_field.setSFVec3f(data_dict["world_info"]["initial_position"])
    rotation_field.setSFRotation(data_dict["world_info"]["initial_rotation"])

    light_translation_field.setSFVec3f(data_dict["world_info"]["light_position"])

    communication = robot.getDevice("emitter")

    while robot.step(timestep) != -1:
        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4096)
        
        rotation_angle = rotation_field.getSFRotation()[3]
        #normalized_rotation_angle = normalize_rotation_angle(rotation_angle)
        sensor_angles = get_sensor_angles(rotation_angle, degree_sensors)
        #best_sensor_idx = np.argmin(normalized_light_sensor_values)

        print(sensor_angles)

        #print(translation_field.getSFVec3f())
        communication.send(json.dumps(translation_field.getSFVec3f()))


        #print(f"[MAIN1] ls{best_sensor_idx} | rotation: {sensor_angles[best_sensor_idx]}")
        #absolute_light_angle = (normalized_rotation_angle + sensor_relative_angle) % (2 * np.pi)
        #communication.send(f"{absolute_light_angle}")

        normalized_sensor_values = torch.tensor(normalized_light_sensor_values)
        directions = robot_network.forward(normalized_sensor_values)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)