"""light_controller controller."""

from controller import Robot, Supervisor
import torch
import numpy as np
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
robot_node = robot.getFromDef("EPUCK")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()

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

    if previous_time > current_time: # nuevo individuo
        current_individual += 1

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        robot.step(timestep)

        robot_node.resetPhysics()

        translation_field.setSFVec3f(INITIAL_POSITION)
        rotation_field.setSFRotation(INITIAL_ROTATION)

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

    # distance_sensor_values = get_sensor_values(distance_sensors)
    # normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 60, 3000)
    
    # input_tensor = torch.hstack([torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)])

    input_tensor = torch.tensor(normalized_light_sensor_values)
    fitness_step = genetic_algorithm.calculate_step_fitness(input_tensor)
    population[current_individual].fitness += fitness_step

    # current_individual_last_position = translation_field.getSFVec3f()

    directions = population[current_individual].network.forward(input_tensor)
    percentage_left_speed = directions[0].item()
    percentage_right_speed = directions[1].item()

    left_motor_velocity = percentage_left_speed * MAX_SPEED
    right_motor_velocity = percentage_right_speed * MAX_SPEED

    #print(f"Left motor velocity: {left_motor_velocity}, Right motor velocity: {right_motor_velocity}, {percentage_left_speed}, {percentage_right_speed}")

    left_motor.setVelocity(left_motor_velocity)
    right_motor.setVelocity(right_motor_velocity)