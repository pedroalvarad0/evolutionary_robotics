"""move_box_controller controller."""

from controller import Robot, Supervisor
import torch
import numpy as np
from genetic_algorithm import GeneticAlgorithm
from robot_network import RobotNetwork
import enum
import torch
import json
import uuid
from utils import read_json_to_dict


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


def save_data(initial_position, initial_rotation, initial_box_position, initial_box_rotation, ga_history):
    # Create a dictionary with world information and individuals data
    data_to_save = {
        "world_info": {
            "initial_position": initial_position,
            "initial_rotation": initial_rotation,
            "initial_box_position": initial_box_position,
            "initial_box_rotation": initial_box_rotation
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
    
    filename = f"ga_history_box_mover_{uuid.uuid4()}.json"
    with open(filename, "w") as f:
        json.dump(data_to_save, f)


def fitness(box_positions, light_finder_positions):
        # Encontrar la longitud mínima entre ambos arreglos
        min_length = min(len(box_positions), len(light_finder_positions))
        
        total_fitness = 0
        
        # Iteramos solo hasta min_length - 1 porque necesitamos t y t-1
        for t in range(1, min_length):
            # Calculamos la distancia en t-1
            d_prev = np.sqrt(
                (box_positions[t-1][0] - light_finder_positions[t-1][0])**2 + 
                (box_positions[t-1][1] - light_finder_positions[t-1][1])**2
            )
            
            # Calculamos la distancia en t
            d_current = np.sqrt(
                (box_positions[t][0] - light_finder_positions[t][0])**2 + 
                (box_positions[t][1] - light_finder_positions[t][1])**2
            )
            
            # Sumamos la diferencia de distancias
            total_fitness += (d_prev - d_current)
        
        return total_fitness


class Mode(enum.Enum):
    TRAINING = 1
    EXECUTION = 2
    TRANSFER_LEARNING = 3


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
MAX_TIME = 60

robot_node = robot.getFromDef("MAIN2")
box_node = robot.getFromDef("BOX")

light_finder_node = robot.getFromDef("MAIN1")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

translation_field_box = box_node.getField("translation")
rotation_field_box = box_node.getField("rotation")

translation_field_light_finder = light_finder_node.getField("translation")
rotation_field_light_finder = light_finder_node.getField("rotation")

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()

INITIAL_POSITION_BOX = translation_field_box.getSFVec3f()
INITIAL_ROTATION_BOX = rotation_field_box.getSFRotation()

INITIAL_POSITION_LIGHT_FINDER = translation_field_light_finder.getSFVec3f()
INITIAL_ROTATION_LIGHT_FINDER = rotation_field_light_finder.getSFRotation()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

distance_sensor_names = ["ps0", "ps1", "ps2", "ps5", "ps6", "ps7"]
distance_sensors = []

for i in range(len(distance_sensor_names)):
    sensor = robot.getDevice(distance_sensor_names[i])
    sensor.enable(timestep)
    distance_sensors.append(sensor)

communication = robot.getDevice("receiver")
communication.enable(timestep)

mode = Mode.EXECUTION

if mode == Mode.TRAINING:

    POPULATION_SIZE = 100
    GENERATIONS = 100

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

    light_finder_positions = []
    box_positions = []

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo
            population[current_individual].fitness = 1000 * fitness(box_positions, light_finder_positions)
            
            #print(f"Generation: {current_generation}, Individual: {current_individual}, Fitness: {population[current_individual].fitness}")

            current_individual += 1

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            robot_node.resetPhysics()
            box_node.resetPhysics()
            light_finder_node.resetPhysics()

            translation_field.setSFVec3f(INITIAL_POSITION)
            rotation_field.setSFRotation(INITIAL_ROTATION)

            translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
            rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

            translation_field_light_finder.setSFVec3f(INITIAL_POSITION_LIGHT_FINDER)
            rotation_field_light_finder.setSFRotation(INITIAL_ROTATION_LIGHT_FINDER)

            if current_individual < POPULATION_SIZE:
                weights_network = population[current_individual].get_weights()
                load_robot_weights(robot_network, weights_network)

            light_finder_positions = []
            box_positions = []
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
                    save_data(INITIAL_POSITION, INITIAL_ROTATION, INITIAL_POSITION_BOX, INITIAL_ROTATION_BOX, ga_history)

        distance_sensor_values = get_sensor_values(distance_sensors)
        normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

        light_finder_x, light_finder_y = 0, 0
        if communication.getQueueLength() > 0:
            box_position = json.loads(communication.getString())
            light_finder_x, light_finder_y = box_position[0], box_position[1]
            light_finder_positions.append([light_finder_x, light_finder_y])
            communication.nextPacket()

        box_position = translation_field_box.getSFVec3f()
        box_positions.append([box_position[0], box_position[1]])

        input_tensor = torch.tensor(normalized_distance_sensor_values)
        input_tensor = torch.cat((input_tensor, torch.tensor([light_finder_x, light_finder_y])))

        directions = robot_network.forward(input_tensor)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED
        
        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)
elif mode == Mode.TRANSFER_LEARNING:
    print("Transfer learning")
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

    data_dict = read_json_to_dict("ga_history_box_mover_60fdd609-4d82-4a47-9a93-1b0aa8c53ec1.json")

    individuals = data_dict["individuals"]

    sorted_individuals = sorted(individuals, key=lambda x: x["fitness"], reverse=True)

    # cargar pesos preentrenados
    for i in range(int(POPULATION_SIZE * 0.80)):
        population[i].weights = sorted_individuals[i]["weights"]

    weights_network = population[0].get_weights()
    load_robot_weights(robot_network, weights_network)

    light_finder_positions = []
    box_positions = []

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo
            population[current_individual].fitness = 1000 * fitness(box_positions, light_finder_positions)
            
            #print(f"Generation: {current_generation}, Individual: {current_individual}, Fitness: {population[current_individual].fitness}")

            current_individual += 1

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            robot_node.resetPhysics()
            box_node.resetPhysics()
            light_finder_node.resetPhysics()

            translation_field.setSFVec3f(INITIAL_POSITION)
            rotation_field.setSFRotation(INITIAL_ROTATION)

            translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
            rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

            translation_field_light_finder.setSFVec3f(INITIAL_POSITION_LIGHT_FINDER)
            rotation_field_light_finder.setSFRotation(INITIAL_ROTATION_LIGHT_FINDER)

            if current_individual < POPULATION_SIZE:
                weights_network = population[current_individual].get_weights()
                load_robot_weights(robot_network, weights_network)

            light_finder_positions = []
            box_positions = []
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
                    save_data(INITIAL_POSITION, INITIAL_ROTATION, INITIAL_POSITION_BOX, INITIAL_ROTATION_BOX, ga_history)

        distance_sensor_values = get_sensor_values(distance_sensors)
        normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

        light_finder_x, light_finder_y = 0, 0
        if communication.getQueueLength() > 0:
            box_position = json.loads(communication.getString())
            light_finder_x, light_finder_y = box_position[0], box_position[1]
            light_finder_positions.append([light_finder_x, light_finder_y])
            communication.nextPacket()

        box_position = translation_field_box.getSFVec3f()
        box_positions.append([box_position[0], box_position[1]])

        input_tensor = torch.tensor(normalized_distance_sensor_values)
        input_tensor = torch.cat((input_tensor, torch.tensor([light_finder_x, light_finder_y])))

        directions = robot_network.forward(input_tensor)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED
        
        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)
elif mode == Mode.EXECUTION:
    file_path = "(3tl)ga_history_box_mover_0affb0d6-0e5c-40e2-baa4-5b9ddfdb6a11.json"
    data_dict = read_json_to_dict(file_path)

    #print(data_dict["individuals"][-2]["fitness"])
    robot_network = RobotNetwork()
    weights_network = data_dict["individuals"][-2]["weights"]
    load_robot_weights(robot_network, weights_network)
    
    translation_field.setSFVec3f(data_dict["world_info"]["initial_position"])
    rotation_field.setSFRotation(data_dict["world_info"]["initial_rotation"])

    translation_field_box.setSFVec3f(data_dict["world_info"]["initial_box_position"])
    rotation_field_box.setSFRotation(data_dict["world_info"]["initial_box_rotation"])

    while robot.step(timestep) != -1:
        distance_sensor_values = get_sensor_values(distance_sensors)
        normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

        light_finder_x, light_finder_y = 0, 0
        if communication.getQueueLength() > 0:
            box_position = json.loads(communication.getString())
            light_finder_x, light_finder_y = box_position[0], box_position[1]
            communication.nextPacket()

        input_tensor = torch.tensor(normalized_distance_sensor_values)
        input_tensor = torch.cat((input_tensor, torch.tensor([light_finder_x, light_finder_y])))

        directions = robot_network.forward(input_tensor)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED
        
        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)
