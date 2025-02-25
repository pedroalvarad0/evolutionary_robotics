"""light_controller controller."""

from controller import Robot, Supervisor
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
from torch.distributions.uniform import Uniform


class RobotNetwork(nn.Module):
    def __init__(self, input_size=16, hidden_size=20):
        super(RobotNetwork, self).__init__()
        # Capa de entrada (8 sensores) a capa oculta
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Capa oculta a capa de salida (2 motores)
        self.fc2 = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # Aplicamos ReLU como función de activación en la capa oculta
        x = F.relu(self.fc1(x))
        # La capa de salida usa tanh para obtener valores entre -1 y 1
        # Esto nos da velocidades positivas y negativas para los motores
        x = torch.tanh(self.fc2(x))
        return x
    

@dataclass
class Individual:
    network: RobotNetwork
    fitness: float


def euclidean_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(list1, list2)))
    

def calculate_step_fitness(normalized_light_sensor_values, normalized_distance_sensor_values, current_position, last_position):

    # Detectar luz (mejor sensor de luz - más cercano a 0 es mejor)
    best_light_sensor = min(normalized_light_sensor_values)
    
    # Detectar obstáculos (peor sensor de proximidad - más cercano a 1 es peor)
    worst_distance_sensor = max(normalized_distance_sensor_values)
    
    # Recompensa por movimiento
    distance_moved = euclidean_distance(current_position, last_position)
    
    # Si no detecta luz (best_light_sensor > 0.7), premiar exploración
    if best_light_sensor > 0.7:
        # Premiar movimiento si no está cerca de obstáculos
        exploration_reward = distance_moved * (1 - worst_distance_sensor)
        fitness = (
            5.0 * exploration_reward -  # Premiar exploración
            10.0 * worst_distance_sensor -  # Penalizar proximidad a obstáculos
            1.0 * sum(normalized_light_sensor_values) / len(normalized_light_sensor_values)  # Pequeña presión para encontrar luz
        )
    else:
        # Si detecta luz, premiar acercamiento a la luz y evitar obstáculos
        light_following_reward = (1 - best_light_sensor) * distance_moved
        fitness = (
            15.0 * (1 - best_light_sensor) +  # Premio fuerte por estar cerca de la luz
            5.0 * light_following_reward -  # Premio por moverse hacia la luz
            10.0 * worst_distance_sensor  # Penalización por acercarse a obstáculos
        )
    
    return fitness

    # best_sensor_value = min(normalized_sensor_values)

    # distance_moved = euclidean_distance(current_position, last_position)

    # movement_reward = distance_moved * (1 - best_sensor_value)

    # fitness = (
    #     10.0 * (1 - best_sensor_value) +
    #     5.0 * movement_reward - 
    #     1.0 * sum(normalized_sensor_values)
    # )

    # return fitness


def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values


def normalize_sensor_values(sensor_values, min_value, max_value):
    normalized = [(x - min_value) / (max_value - min_value) for x in sensor_values]
    return normalized


def tournament_selection(population, tournament_size=3):
    tournament = list(np.random.choice(population, tournament_size))
    tournament.sort(key=lambda x: x.fitness, reverse=True)
    return tournament[0]


def crossover(parent1, parent2):
    beta_distribution = Uniform(-0.25, 1.25)

    beta_matrix_fc1_weight = beta_distribution.sample(parent1.network.fc1.weight.shape)
    beta_matrix_fc1_bias = beta_distribution.sample(parent1.network.fc1.bias.shape)

    beta_matrix_fc2 = beta_distribution.sample(parent1.network.fc2.weight.shape)
    beta_matrix_fc2_bias = beta_distribution.sample(parent1.network.fc2.bias.shape)

    
    new_fc1_weight = parent1.network.fc1.weight.data * beta_matrix_fc1_weight + parent2.network.fc1.weight.data * (1 - beta_matrix_fc1_weight)
    new_fc1_bias = parent1.network.fc1.bias.data * beta_matrix_fc1_bias + parent2.network.fc1.bias.data * (1 - beta_matrix_fc1_bias)

    new_fc2_weight = parent1.network.fc2.weight.data * beta_matrix_fc2 + parent2.network.fc2.weight.data * (1 - beta_matrix_fc2)
    new_fc2_bias = parent1.network.fc2.bias.data * beta_matrix_fc2_bias + parent2.network.fc2.bias.data * (1 - beta_matrix_fc2_bias)
    
    child_network = RobotNetwork()
    child_network.fc1.weight.data = new_fc1_weight
    child_network.fc1.bias.data = new_fc1_bias
    child_network.fc2.weight.data = new_fc2_weight
    child_network.fc2.bias.data = new_fc2_bias

    child = Individual(network=child_network, fitness=0.0)

    return child


def mutate(individual):
    std = (1 - (-1)) / 6
    mutation_matrix_fc1_weight = torch.randn(individual.network.fc1.weight.shape) * std
    mutation_matrix_fc1_bias = torch.randn(individual.network.fc1.bias.shape) * std
    mutation_matrix_fc2_weight = torch.randn(individual.network.fc2.weight.shape) * std
    mutation_matrix_fc2_bias = torch.randn(individual.network.fc2.bias.shape) * std

    new_fc1_weight = individual.network.fc1.weight.data + mutation_matrix_fc1_weight
    new_fc1_bias = individual.network.fc1.bias.data + mutation_matrix_fc1_bias
    new_fc2_weight = individual.network.fc2.weight.data + mutation_matrix_fc2_weight
    new_fc2_bias = individual.network.fc2.bias.data + mutation_matrix_fc2_bias

    individual.network.fc1.weight.data = new_fc1_weight
    individual.network.fc1.bias.data = new_fc1_bias
    individual.network.fc2.weight.data = new_fc2_weight
    individual.network.fc2.bias.data = new_fc2_bias

    return individual


def create_next_generation(population, crossover_rate=0.8, mutation_rate=0.4):
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    fittest_individual = population[0]

    new_population = []

    while len(new_population) < len(population):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        if np.random.random() < crossover_rate:
            child = crossover(parent1, parent2)
        else:
            child = np.random.choice([parent1, parent2])

        if np.random.random() < mutation_rate:
            child = mutate(child)

        new_population.append(child)

    return fittest_individual, new_population


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
population = [Individual(network=RobotNetwork(), fitness=0.0) for _ in range(POPULATION_SIZE)]
history = []

while robot.step(timestep) != -1:
    
    previous_time = current_time
    current_time = robot.getTime() % MAX_TIME

    if previous_time > current_time: # nuevo individuo
        #print(current_individual, population[current_individual].fitness)

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
            fittest_individual, new_population = create_next_generation(population)

            print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")

            history.append(fittest_individual.fitness)
            population = new_population
            current_generation += 1
            if current_generation == GENERATIONS:
                pass # TODO: save best individual
    
    sensor_values = get_sensor_values(light_sensors)
    normalized_light_sensor_values = normalize_sensor_values(sensor_values, 0, 4095)

    distance_sensor_values = get_sensor_values(distance_sensors)
    normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 60, 3000)

    #print(normalized_distance_sensor_values)
    
    input_tensor = torch.hstack([torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)])

    fitness_step = calculate_step_fitness(normalized_light_sensor_values, normalized_distance_sensor_values, translation_field.getSFVec3f(), current_individual_last_position)
    population[current_individual].fitness += fitness_step

    current_individual_last_position = translation_field.getSFVec3f()

    directions = population[current_individual].network.forward(input_tensor)
    percentage_left_speed = directions[0].item()
    percentage_right_speed = directions[1].item()

    left_motor.setVelocity(percentage_left_speed * MAX_SPEED)
    right_motor.setVelocity(percentage_right_speed * MAX_SPEED)

    # if robot.getTime() % 60 == 0:
    #     print(f"Generation: {current_generation}")

    # print(f"Generation: {current_generation}, Individual: {current_individual}, Fitness: {population[current_individual].fitness}")