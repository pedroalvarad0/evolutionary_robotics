"""light_controller controller."""

from controller import Robot, Supervisor
import torch
import numpy as np
import random
#from genetic_algorithm import Individual, create_next_generation, calculate_step_fitness
from genetic_algorithm import GeneticAlgorithm
from robot_network import RobotNetwork


def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values


def normalize_sensor_values(sensor_values, min_value, max_value):
    normalized = [(x - min_value) / (max_value - min_value) for x in sensor_values]
    return normalized


MAX_SPEED = 6.28
MAX_TIME = 30
POPULATION_SIZE = 100
GENERATIONS = 20

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# nodes
robot_node = robot.getFromDef("EPUCK")
light_node = robot.getFromDef("LIGHT")
arena_node = robot.getFromDef("ARENA")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

light_translation_field = light_node.getField("translation")
arena_translation_field = arena_node.getField("translation")
arena_translation = arena_translation_field.getSFVec3f()
arena_size = 1.0 # 1x1 metros

ARENA_LIMITS = {
    'x_min': arena_translation[0] - arena_size/2,
    'x_max': arena_translation[0] + arena_size/2,
    'y_min': arena_translation[2] - arena_size/2,
    'y_max': arena_translation[2] + arena_size/2,
    'z': 0.25  # altura fija de la luz
}

LIGHT_MOVE_INTERVAL = 5.0
last_light_move_time = 0

LIGHT_STEP_SIZE = 0.0015 # Tamaño del paso en cada movimiento
LIGHT_DIRECTION_CHANGE_INTERVAL = 3.0 # Cada cuántos segundos cambia de dirección
last_direction_change = 0
light_direction = [random.uniform(-1, 1), random.uniform(-1, 1), 0] # Dirección inicial aleatoria

def normalize_direction(direction):
    """Normaliza un vector de dirección"""
    magnitude = (direction[0]**2 + direction[1]**2)**0.5
    if magnitude == 0:
        return [1, 0, 0]
    return [direction[0]/magnitude, direction[1]/magnitude, 0]

def move_light_step(light_translation_field):
    """Mueve la luz un paso en la dirección actual, respetando los límites"""
    current_pos = light_translation_field.getSFVec3f()
    
    # Calcular nueva posición
    new_x = current_pos[0] + light_direction[0] * LIGHT_STEP_SIZE
    new_y = current_pos[1] + light_direction[1] * LIGHT_STEP_SIZE
    
    # Verificar y ajustar límites
    if new_x < ARENA_LIMITS['x_min']:
        new_x = ARENA_LIMITS['x_min']
        light_direction[0] *= -1  # Rebota en la pared
    elif new_x > ARENA_LIMITS['x_max']:
        new_x = ARENA_LIMITS['x_max']
        light_direction[0] *= -1  # Rebota en la pared
        
    if new_y < ARENA_LIMITS['y_min']:
        new_y = ARENA_LIMITS['y_min']
        light_direction[1] *= -1  # Rebota en la pared
    elif new_y > ARENA_LIMITS['y_max']:
        new_y = ARENA_LIMITS['y_max']
        light_direction[1] *= -1  # Rebota en la pared
    
    new_position = [new_x, new_y, ARENA_LIMITS['z']]
    light_translation_field.setSFVec3f(new_position)

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()
INITIAL_LIGHT_POSITION = light_translation_field.getSFVec3f()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]
light_sensors = []

distance_sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
distance_sensors = []

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

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
    crossover_rate=0.8,
    mutation_rate=0.02,
    representation="binary"
)

population = genetic_algorithm.generate_initial_population()
history = []

while robot.step(timestep) != -1:
    
    previous_time = current_time
    current_time = robot.getTime() % MAX_TIME

    # if current_time - last_light_move_time >= LIGHT_MOVE_INTERVAL:
    #     light_translation_field.setSFVec3f([random.uniform(ARENA_LIMITS['x_min'], ARENA_LIMITS['x_max']), random.uniform(ARENA_LIMITS['y_min'], ARENA_LIMITS['y_max']), ARENA_LIMITS['z']])
    #     last_light_move_time = current_time

    if current_time - last_direction_change >= LIGHT_DIRECTION_CHANGE_INTERVAL:
        light_direction = normalize_direction([random.uniform(-1, 1), random.uniform(-1, 1), 0])
        last_direction_change = current_time

    move_light_step(light_translation_field)

    if previous_time > current_time: # nuevo individuo
        current_individual += 1

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        robot.step(timestep)

        robot_node.resetPhysics()

        translation_field.setSFVec3f(INITIAL_POSITION)
        rotation_field.setSFRotation(INITIAL_ROTATION)
        light_translation_field.setSFVec3f(INITIAL_LIGHT_POSITION)
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
    
    sensor_values = get_sensor_values(light_sensors)
    normalized_light_sensor_values = normalize_sensor_values(sensor_values, 0, 4095)

    #print(normalized_light_sensor_values)
    #print(light_translation_field.getSFVec3f())

    input_tensor = torch.tensor(normalized_light_sensor_values)
    fitness_step = genetic_algorithm.calculate_step_fitness(input_tensor)
    population[current_individual].fitness += fitness_step

    directions = population[current_individual].network.forward(input_tensor)
    percentage_left_speed = directions[0].item()
    percentage_right_speed = directions[1].item()

    left_motor_velocity = percentage_left_speed * MAX_SPEED
    right_motor_velocity = percentage_right_speed * MAX_SPEED

    left_motor.setVelocity(left_motor_velocity)
    right_motor.setVelocity(right_motor_velocity)