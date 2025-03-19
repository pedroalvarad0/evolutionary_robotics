"""main_controller controller."""
from controller import Robot, Supervisor
from genetic_algorithm import GeneticAlgorithm, fitness, move_box_fitness
import numpy as np
import torch
import json
from utils import get_sensor_values,normalize_sensor_values, load_robot_weights, create_config_file, save_generation_data, read_json_to_dict, get_np_image_from_camera, calculate_average_color, get_history_info, get_last_generation_info
from robot_network import RobotNetwork, SimpleRobotNetwork
import uuid
import enum
import time

def reset_positions(fields, initial_positions):
    fields["robot1"]["translation"].setSFVec3f(initial_positions["robot1"]["translation"])
    fields["robot1"]["rotation"].setSFRotation(initial_positions["robot1"]["rotation"])

    fields["robot2"]["translation"].setSFVec3f(initial_positions["robot2"]["translation"])
    fields["robot2"]["rotation"].setSFRotation(initial_positions["robot2"]["rotation"])
    
    fields["object"]["translation"].setSFVec3f(initial_positions["object"]["translation"])
    fields["object"]["rotation"].setSFRotation(initial_positions["object"]["rotation"])

    #fields["area"]["translation"].setSFVec3f(initial_positions["area"]["translation"])


def reset_physics(robot1_node, robot2_node, object_node):
    robot1_node.resetPhysics()
    robot2_node.resetPhysics()
    object_node.resetPhysics()


class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    CONTINUE_TRAINING = 2   # Mode for continuing the training
    EXECUTION = 3   # Mode for using the trained controller

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep() / 2)

MAX_SPEED = 6.28
MAX_TIME = 60
POPULATION_SIZE = 25
GENERATIONS = 500
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
REPRESENTATION = "real"

MAX_SPEED = 6.28
max_time = 60
population_size = 25
generations = 500
crossover_rate = 0.9
mutation_rate = 0.1
representation = "real"

# get robot devices
camera = robot.getDevice("camera")
camera.enable(timestep)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

light_sensors = []
light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

# get world nodes
robot1_node = robot.getFromDef("ROBOT1")
robot2_node = robot.getFromDef("ROBOT2")

#box_node = robot.getFromDef("BOX")
object_node = robot.getFromDef("OBJECT")
area_node = robot.getFromDef("AREA")

translation_field_robot1 = robot1_node.getField("translation")
rotation_field_robot1 = robot1_node.getField("rotation")

translation_field_robot2 = robot2_node.getField("translation")
rotation_field_robot2 = robot2_node.getField("rotation")

translation_field_object = object_node.getField("translation")
rotation_field_object = object_node.getField("rotation")

translation_field_area = area_node.getField("translation")

custom_data_field_robot1 = robot1_node.getField("customData")
custom_data_field_robot2 = robot2_node.getField("customData")

fields = {
    "robot1": {
        "translation": translation_field_robot1,
        "rotation": rotation_field_robot1
    },
    "robot2": {
        "translation": translation_field_robot2,
        "rotation": rotation_field_robot2
    },
    "object": {
        "translation": translation_field_object,
        "rotation": rotation_field_object
    },
    "area": {
        "translation": translation_field_area
    }
}

initial_positions = {
    "robot1": {
        "translation": translation_field_robot1.getSFVec3f(),
        "rotation": rotation_field_robot1.getSFRotation()
    },
    "robot2": {
        "translation": translation_field_robot2.getSFVec3f(),
        "rotation": rotation_field_robot2.getSFRotation()
    },
    "object": {
        "translation": translation_field_object.getSFVec3f(),
        "rotation": rotation_field_object.getSFRotation()
    },
    "area": {
        "translation": translation_field_area.getSFVec3f()
    }
}

mode = Mode.EXECUTION

if robot_name == "robot1":
    ga_uuid = ""
    current_individual = 0
    current_generation = 0

    if mode == Mode.TRAINING:
        ga_uuid = uuid.uuid4()

        create_config_file(
            ga_uuid,
            max_time,
            population_size,
            generations,
            crossover_rate,
            mutation_rate,
            representation,
            initial_positions["robot1"]["translation"],
            initial_positions["robot1"]["rotation"],
            initial_positions["robot2"]["translation"],
            initial_positions["robot2"]["rotation"],
            initial_positions["object"]["translation"],
            initial_positions["object"]["rotation"],
        )
    elif mode == Mode.CONTINUE_TRAINING:
        ga_uuid = "11cf44b2-e3e8-49e5-a806-ede387de9328"
        config, gens_info = get_history_info(ga_uuid)
        last_generation_info = get_last_generation_info(ga_uuid)

        current_individual = 0
        current_generation = last_generation_info["generation"]

        max_time = config["max_time"]
        population_size = config["population_size"]
        generations = config["generations"]
        crossover_rate = config["crossover_rate"]
        mutation_rate = config["mutation_rate"]
        representation = config["representation"]

        initial_positions["robot1"]["translation"] = config["initial_position_robot1"]
        initial_positions["robot1"]["rotation"] = config["initial_rotation_robot1"]
        initial_positions["robot2"]["translation"] = config["initial_position_robot2"]
        initial_positions["robot2"]["rotation"] = config["initial_rotation_robot2"]
        initial_positions["object"]["translation"] = config["initial_position_object"]
        initial_positions["object"]["rotation"] = config["initial_rotation_object"]

        print(f"[CONTINUE TRAINING] Reanudando entrenamiento de {ga_uuid}. Generation: {current_generation}")

    current_time = 0
    previous_time = 0

    robot_network = SimpleRobotNetwork()

    genetic_algorithm = GeneticAlgorithm(
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        representation=representation
    )
    population = genetic_algorithm.generate_initial_population()
    initial_weights = population[0].get_weights()
    load_robot_weights(robot_network, initial_weights)

    if mode == Mode.EXECUTION:
        ga_uuid = "1b9cda21-241b-49a8-a4c0-5d56d3c86a1f"
        generation_file = "generation_25.json"

        generation_data = read_json_to_dict(f"histories/{ga_uuid}/{generation_file}")
        
        print(f"[EXECUTION]Generation: {generation_data['generation']}, Best Fitness: {generation_data['fittest_individual_fitness']}")

        weights = generation_data["fittest_individual_weights"]
        load_robot_weights(robot_network, weights)

    current_time = 0
    previous_time = 0

    while robot.step(timestep) != -1:

        if mode == Mode.TRAINING or mode == Mode.CONTINUE_TRAINING:
            previous_time = current_time
            current_time = robot.getTime() % max_time

            if previous_time > current_time:
                population[current_individual].fitness = fitness(fields["object"]["translation"].getSFVec3f(), initial_positions["object"]["translation"])

                current_individual += 1

                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                reset_physics(robot1_node, robot2_node, object_node)
                reset_positions(fields, initial_positions)

                if current_individual < POPULATION_SIZE:
                    weights = population[current_individual].get_weights()
                    load_robot_weights(robot_network, weights)
                    
                if current_individual == POPULATION_SIZE:
                    current_individual = 0
                    fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                    print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")
                    save_generation_data(fittest_individual, population, current_generation, ga_uuid)

                    population = new_population
                    current_generation += 1

                    next_weights = population[0].get_weights()
                    load_robot_weights(robot_network, next_weights)

                    if current_generation == generations:
                        pass
        
        robot2_inputs = torch.zeros(9)
        if receiver.getQueueLength() > 0:
            robot2_inputs = torch.tensor(json.loads(receiver.getString()))
            receiver.nextPacket()

        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
        discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

        image = get_np_image_from_camera(camera)
        average_color = calculate_average_color(image).tolist()
        seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)
        self_inputs = torch.cat((torch.tensor(discretize_light_sensor_values).to(torch.float32), torch.tensor([seeing_object]).to(torch.float32)))

        inputs = torch.cat((self_inputs, robot2_inputs))
        
        outputs = robot_network.forward(inputs)
        percentage_left_speed = outputs[0].item()
        percentage_right_speed = outputs[1].item()

        robot2_percentage_left_speed = outputs[2].item()
        robot2_percentage_right_speed = outputs[3].item()

        emitter.send(str([robot2_percentage_left_speed, robot2_percentage_right_speed]))

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)
elif robot_name == "robot2":
    current_time = 0
    previous_time = 0

    while robot.step(timestep) != -1:

        if mode == Mode.TRAINING or mode == Mode.CONTINUE_TRAINING:
            previous_time = current_time
            current_time = robot.getTime() % max_time

            if previous_time > current_time:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
        discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

        image = get_np_image_from_camera(camera)
        average_color = calculate_average_color(image).tolist()
        seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)
        emitter.send(str(discretize_light_sensor_values + [seeing_object]))

        directions = [0, 0]
        if receiver.getQueueLength() > 0:
            directions = json.loads(receiver.getString())
            receiver.nextPacket()

        left_motor_velocity = directions[0] * MAX_SPEED
        right_motor_velocity = directions[1] * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)