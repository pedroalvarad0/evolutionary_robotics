"""general_move_box_controller controller."""
import torch
from controller import Supervisor
from genetic_algorithm import GeneticAlgorithm, fitness, move_object_fitness
from robot_network import SimpleRobotNetwork, RobotNetwork
from utils import (
    load_robot_weights,
    get_sensor_values, 
    normalize_sensor_values, 
    get_np_image_from_camera, 
    calculate_average_color, 
    AreaSampler, 
    create_config_file, 
    save_generation_data, 
    read_json_to_dict, 
    get_history_info, 
    get_last_generation_info
)
import numpy as np
import json
import uuid
import enum

def generate_random_epuck_positions(x_min, x_max, y_min, y_max):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    rotation = np.random.uniform(-np.pi, np.pi)
    return x, y, rotation


def generation_random_object_position(arena_position, sigma_x, sigma_y, x_min, x_max, y_min, y_max):
    x = np.random.normal(loc=arena_position[0], scale=sigma_x)
    y = np.random.normal(loc=arena_position[1], scale=sigma_y)
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    rotation = np.random.uniform(-np.pi, np.pi)
    return x, y, rotation


def generate_random_initial_positions(n, area_position, area_size):
    """
    Genera n configuraciones válidas de posiciones iniciales
    """
    sampler = AreaSampler(center=area_position, size=area_size, margin=0.15, min_distance=0.2)
    positions = []
    
    while len(positions) < n:
        config = sampler.generate_valid_configuration()
        if config is not None:
            positions.append(config)
    
    return positions

def set_initial_positions(positions, fields, nodes):
    nodes["epuck1"].resetPhysics()
    nodes["epuck2"].resetPhysics()
    nodes["object"].resetPhysics()

    epuck1_translation = fields["epuck1"]["translation"].getSFVec3f()
    epuck1_rotation = fields["epuck1"]["rotation"].getSFRotation()

    epuck2_translation = fields["epuck2"]["translation"].getSFVec3f()
    #epuck2_rotation = fields["epuck2"]["rotation"].getSFRotation()

    object_translation = fields["object"]["translation"].getSFVec3f()
    object_rotation = fields["object"]["rotation"].getSFRotation()  
    
    epuck1_translation[0] = positions["epuck1"][0]
    epuck1_translation[1] = positions["epuck1"][1]
    epuck1_rotation[-1] = positions["epuck1"][2]

    epuck2_translation[0] = positions["epuck2"][0]
    epuck2_translation[1] = positions["epuck2"][1]
    #epuck2_rotation[-1] = positions["epuck2"][2]
    
    object_translation[0] = positions["object"][0]
    object_translation[1] = positions["object"][1]
    object_rotation[-1] = positions["object"][2]

    fields["epuck1"]["translation"].setSFVec3f(epuck1_translation)
    fields["epuck2"]["translation"].setSFVec3f(epuck2_translation)
    fields["object"]["translation"].setSFVec3f(object_translation)

    fields["epuck1"]["rotation"].setSFRotation(epuck1_rotation)
    fields["epuck2"]["rotation"].setSFRotation([0, 0, 1, positions["epuck2"][2]])
    fields["object"]["rotation"].setSFRotation(object_rotation)


class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    CONTINUE_TRAINING = 2   # Mode for continuing the training
    EXECUTION = 3   # Mode for using the trained controller

MAX_SPEED = 6.28

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep())

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

# world objects, node and fields

nodes = {
    "epuck1": robot.getFromDef("EPUCK1"),
    "epuck2": robot.getFromDef("EPUCK2"),
    "object": robot.getFromDef("OBJECT"),
    "arena": robot.getFromDef("ARENA")
}

fields = {
    "epuck1": {
        "translation": nodes["epuck1"].getField("translation"),
        "rotation": nodes["epuck1"].getField("rotation")
    },
    "epuck2": {
        "translation": nodes["epuck2"].getField("translation"),
        "rotation": nodes["epuck2"].getField("rotation")
    },
    "object": {
        "translation": nodes["object"].getField("translation"),
        "rotation": nodes["object"].getField("rotation")
    },
    "arena": {
        "translation": nodes["arena"].getField("translation"),
        "floorSize": nodes["arena"].getField("floorSize")
    }
}

# simulation and training parameters
max_time = 30
tests_per_individual = 5
current_test = 0
random_initial_positions = generate_random_initial_positions(tests_per_individual, fields["arena"]["translation"].getSFVec3f(), fields["arena"]["floorSize"].getSFVec2f())

# Genetic Algorithm
population_size = 50
crossover_rate = 0.9
mutation_rate = 0.01
representation = "real"
generations = 100

reset_initial_conditions_every_n_generations = 10

genetic_algorithm = GeneticAlgorithm(
    population_size=population_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    representation=representation
)

current_individual = 0
current_generation = 0

population = genetic_algorithm.generate_initial_population()

# Robot Network configurations
robot_network = RobotNetwork()
initial_weights = population[0].get_weights()
load_robot_weights(robot_network, initial_weights)

current_time = 0
previous_time = 0

move_object_history = []
inputs_epuck1 = []
inputs_epuck2 = []

set_initial_positions(random_initial_positions[current_test], fields, nodes)

mode = Mode.TRAINING

ga_uuid = ""

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
        tests_per_individual
    )
elif mode == Mode.CONTINUE_TRAINING:
    ga_uuid = "078c7ee4-05e7-43e2-95cd-3f9a49e2ca3a"
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
    tests_per_individual = config["tests_per_individual"]

    for individual in population:
        individual.weights = last_generation_info["population"][current_individual]["weights"]
    
elif mode == Mode.EXECUTION:
    ga_uuid = "078c7ee4-05e7-43e2-95cd-3f9a49e2ca3a"
    generation_file = "generation_120.json"

    generation_data = read_json_to_dict(f"histories/{ga_uuid}/{generation_file}")
    
    print(f"[EXECUTION] Generation: {generation_data['generation']}, Best Fitness: {generation_data['fittest_individual_fitness']}")

    weights = generation_data["fittest_individual_weights"]
    load_robot_weights(robot_network, weights)

while robot.step(timestep) != -1:
    previous_time = current_time
    current_time = robot.getTime() % max_time

    if mode == Mode.TRAINING or mode == Mode.CONTINUE_TRAINING:
        if previous_time > current_time:
            # calculate fitness
            #population[current_individual].fitness = fitness(move_object_history, inputs_epuck1, inputs_epuck2)
            population[current_individual].fitness += move_object_fitness(move_object_history)
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            move_object_history = []
            # inputs_epuck1 = []
            # inputs_epuck2 = []

            current_test += 1

            if current_test < tests_per_individual: # same individual, different initial positions
                set_initial_positions(random_initial_positions[current_test], fields, nodes)
            else: # new individual, same generation
                current_test = 0
                current_individual += 1

                if current_individual < population_size: # new individual, same generation
                    weights = population[current_individual].get_weights()
                    load_robot_weights(robot_network, weights)
                    
                    set_initial_positions(random_initial_positions[current_test], fields, nodes)
                else: # new generation
                    current_individual = 0

                    for individual in population:
                        individual.fitness = individual.fitness / tests_per_individual

                    fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                    print(f"Generation {current_generation}, Best Fitness: {fittest_individual.fitness}")

                    save_generation_data(fittest_individual, population, current_generation, ga_uuid)

                    population = new_population
                    current_generation += 1

                    next_weights = population[0].get_weights()
                    load_robot_weights(robot_network, next_weights)

                    if current_generation % reset_initial_conditions_every_n_generations == 0:
                        random_initial_positions = generate_random_initial_positions(tests_per_individual, fields["arena"]["translation"].getSFVec3f(), fields["arena"]["floorSize"].getSFVec2f())

                    set_initial_positions(random_initial_positions[current_test], fields, nodes)

                    if current_generation == generations:
                        pass

    move_object_history.append(fields["object"]["translation"].getSFVec3f())

    epuck2_inputs = torch.zeros(9)
    if receiver.getQueueLength() > 0:
        epuck2_inputs = torch.tensor(json.loads(receiver.getString()))
        receiver.nextPacket()

    epuck2_inputs = epuck2_inputs.to(torch.float32)

    #inputs_epuck2.append(epuck2_inputs.tolist())

    light_sensor_values = get_sensor_values(light_sensors)
    normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
    discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

    image = get_np_image_from_camera(camera)
    average_color = calculate_average_color(image).tolist()
    seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)

    inputs = torch.cat((torch.tensor(discretize_light_sensor_values), torch.tensor([seeing_object]))).to(torch.float32)
    #inputs_epuck1.append(inputs.tolist())

    outputs = robot_network.forward(inputs)
    outputs_epuck2 = robot_network.forward(epuck2_inputs)

    percentage_left_speed = outputs[0].item()
    percentage_right_speed = outputs[1].item()

    percentage_left_speed_epuck2 = outputs_epuck2[0].item()
    percentage_right_speed_epuck2 = outputs_epuck2[1].item()

    emitter.send(str([percentage_left_speed_epuck2, percentage_right_speed_epuck2]))

    left_motor.setVelocity(percentage_left_speed * MAX_SPEED)
    right_motor.setVelocity(percentage_right_speed * MAX_SPEED)