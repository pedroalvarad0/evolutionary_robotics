import torch
import numpy as np
from dataclasses import dataclass
from robot_network import RobotNetwork
from torch.distributions.uniform import Uniform

@dataclass
class Individual:
    network: RobotNetwork
    fitness: float


class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=100, crossover_rate=0.8, mutation_rate=0.02, representation="binary"):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.representation = representation

    def generate_random_individual(self):
        return Individual(network=RobotNetwork(), fitness=0.0)
    
    def generate_initial_population(self):
        return [self.generate_random_individual() for _ in range(self.population_size)]
    
    def calculate_step_fitness(self, normalized_light_sensor_values):
        # Inicializamos el fitness en 0
        fitness = 0.0
        
        # Por cada sensor, si está recibiendo luz (valor cercano a 0)
        # incrementamos el fitness
        for sensor_value in normalized_light_sensor_values:
            # Como 0 es máxima luz y 1 es mínima luz,
            # restamos el valor de 1 para que valores bajos den más fitness
            fitness += (1 - sensor_value)
        
        return fitness
    
    def create_next_generation(self, population):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        fittest_individual = population[0]

        new_population = []

        while len(new_population) < len(population):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            if np.random.random() < self.crossover_rate:
                if self.representation == "real":
                    child = self.crossover_real_valued(parent1, parent2)
                elif self.representation == "binary":
                    child = self.crossover_binary(parent1, parent2)
            else:
                child = np.random.choice([parent1, parent2])

            if np.random.random() < self.mutation_rate:
                if self.representation == "real":
                    child = self.mutate_real_valued(child)
                elif self.representation == "binary":
                    child = self.mutate_binary(child)
	
            new_population.append(child)

        return fittest_individual, new_population
    
    def tournament_selection(self, population, tournament_size=3):
        tournament = list(np.random.choice(population, tournament_size))
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def crossover_real_valued(self, parent1, parent2):
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
    
    def crossover_binary(self, parent1, parent2):
        pass
    
    def mutate_real_valued(self, individual):
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
    
    def mutate_binary(self, individual):
        pass