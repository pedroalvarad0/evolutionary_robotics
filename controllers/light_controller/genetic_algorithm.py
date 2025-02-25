import torch
import numpy as np
from dataclasses import dataclass
from robot_network import RobotNetwork
from torch.distributions.uniform import Uniform

@dataclass
class Individual:
    network: RobotNetwork
    fitness: float


def calculate_step_fitness(normalized_light_sensor_values):
    # Inicializamos el fitness en 0
    fitness = 0.0
    
    # Por cada sensor, si está recibiendo luz (valor cercano a 0)
    # incrementamos el fitness
    for sensor_value in normalized_light_sensor_values:
        # Como 0 es máxima luz y 1 es mínima luz,
        # restamos el valor de 1 para que valores bajos den más fitness
        fitness += (1 - sensor_value)
    
    return fitness


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