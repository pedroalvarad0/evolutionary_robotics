"""move_box_controller controller."""

from controller import Robot, Supervisor
import torch
import numpy as np
from genetic_algorithm import GeneticAlgorithm

def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values

def normalize_sensor_values(sensor_values, min_value, max_value):
    normalized = [(x - min_value) / (max_value - min_value) for x in sensor_values]
    return normalized

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getFromDef("EPUCK")
box_node = robot.getFromDef("BOX")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

translation_field_box = box_node.getField("translation")
rotation_field_box = box_node.getField("rotation")

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()

INITIAL_POSITION_BOX = translation_field_box.getSFVec3f()
INITIAL_ROTATION_BOX = rotation_field_box.getSFRotation()

MAX_SPEED = 6.28
MAX_TIME = 30
POPULATION_SIZE = 100
GENERATIONS = 20

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

distance_sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
distance_sensors = []

for i in range(len(distance_sensor_names)):
    sensor = robot.getDevice(distance_sensor_names[i])
    sensor.enable(timestep)
    distance_sensors.append(sensor)

current_individual_last_position = translation_field.getSFVec3f()
current_individual = 0
current_generation = 0
current_time = 0
previous_time = 0

genetic_algorithm = GeneticAlgorithm(
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    crossover_rate=0.9,
    mutation_rate=0.05,
    representation="binary"
)

box_positions_history = []

population = genetic_algorithm.generate_initial_population()
history = []

# def fitness(initial_box_position, final_box_position):
#     distance = 1000 * np.sqrt((final_box_position[0] - initial_box_position[0])**2 + (final_box_position[1] - initial_box_position[1])**2)
#     return distance

def fitness_box_positions(box_positions_history):
    total_distance = 0
    for i in range(1, len(box_positions_history)):
        # Calcular la distancia entre la posición actual y la anterior
        x_current, y_current = box_positions_history[i][0], box_positions_history[i][1]
        x_prev, y_prev = box_positions_history[i-1][0], box_positions_history[i-1][1]
        
        # Aplicar la fórmula de distancia euclidiana
        distance = np.sqrt((x_current - x_prev)**2 + (y_current - y_prev)**2)
        total_distance += distance
    
    return 1000 * total_distance
    

while robot.step(timestep) != -1:
    previous_time = current_time
    current_time = robot.getTime() % MAX_TIME

    if previous_time > current_time: # nuevo individuo
        #population[current_individual].fitness = genetic_algorithm.calculate_fitness(l0_value_history)
        #population[current_individual].fitness = fitness(translation_field.getSFVec3f(), LIGHT_POSITION)
        #population[current_individual].fitness = fitness(INITIAL_POSITION_BOX, translation_field_box.getSFVec3f())
        population[current_individual].fitness = fitness_box_positions(box_positions_history)

        current_individual += 1

        box_positions_history = []

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        robot.step(timestep)

        robot_node.resetPhysics()
        box_node.resetPhysics()

        translation_field.setSFVec3f(INITIAL_POSITION)
        rotation_field.setSFRotation(INITIAL_ROTATION)

        translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
        rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

        robot.step(timestep)

        if current_individual == POPULATION_SIZE:
            current_individual = 0

            # next generation
            fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

            print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")

            history.append(fittest_individual.fitness)
            population = new_population
            current_generation += 1
            if current_generation == GENERATIONS:
                pass # TODO: save best individual

    box_positions_history.append(translation_field_box.getSFVec3f())
    
    sensor_values = get_sensor_values(distance_sensors)
    normalized_distance_sensor_values = normalize_sensor_values(sensor_values, 0, 1000)
    
    input_tensor = torch.tensor(normalized_distance_sensor_values)

    directions = population[current_individual].network.forward(input_tensor)
    percentage_left_speed = directions[0].item()
    percentage_right_speed = directions[1].item()

    left_motor_velocity = percentage_left_speed * MAX_SPEED
    right_motor_velocity = percentage_right_speed * MAX_SPEED

    left_motor.setVelocity(left_motor_velocity)
    right_motor.setVelocity(right_motor_velocity)