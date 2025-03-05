import torch
import numpy as np
from dataclasses import dataclass
from robot_network import RobotNetwork
from torch.distributions.uniform import Uniform
import struct

class Individual:
    def __init__(self, network):
        self.fitness = 0.0
        self.network = network
        self.binary_weights = self.weights_to_binary()

    def weights_to_binary(self):
        binary_representation = []
        for param in self.network.parameters():
            weights = param.data.numpy().flatten()
            for weight in weights:
                binary = format(struct.unpack('!I', struct.pack('!f', weight))[0], '032b')
                binary_representation.extend(list(binary))
        return binary_representation

    def binary_to_weights(self):
        weights = []
        for i in range(0, len(self.binary_weights), 32):
            binary_weight = ''.join(self.binary_weights[i:i+32])
            float_weight = struct.unpack('!f', struct.pack('!I', int(binary_weight, 2)))[0]
            weights.append(float_weight)
        return weights
    
    def update_network_weights(self):
        weights = self.binary_to_weights()
        idx = 0
        for param in self.network.parameters():
            layer_size = param.data.numel()
            layer_weights = weights[idx:idx + layer_size]
            param.data = torch.tensor(layer_weights).reshape(param.data.shape)
            idx += layer_size

class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=100, crossover_rate=0.8, mutation_rate=0.02, representation="binary"):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.representation = representation

    def generate_random_individual(self):
        return Individual(network=RobotNetwork())
    
    def generate_initial_population(self):
        return [self.generate_random_individual() for _ in range(self.population_size)]
    
    def calculate_step_fitness(self, normalized_light_sensor_values):
        best_sensor_value = min(normalized_light_sensor_values)

        idx = np.argmin(normalized_light_sensor_values)

        #print(f"ls{idx} detectando mayor valor de luz {best_sensor_value}")
        
        return 1 - best_sensor_value
    
    def calculate_fitness(self, best_sensor_value_history):
        sensor_data = np.array(best_sensor_value_history)
        fitness_step = 1 - sensor_data
        return np.sum(fitness_step)
    
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
                    self.mutate_real_valued(child)

                    new_population.append(child)
                    
                elif self.representation == "binary":
                    child1, child2 = self.crossover_binary(parent1, parent2)

                    self.mutate_binary(child1)
                    self.mutate_binary(child2)

                    new_population.append(child1)
                    new_population.append(child2)
            else:
                child = np.random.choice([parent1, parent2])
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

        child = Individual(network=child_network)

        return child
    
    def two_point_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32
        # Ensure points are aligned with 32-bit boundaries and point1 < point2
        point1 = np.random.randint(0, num_weights - 1) * 32
        point2 = np.random.randint(point1//32 + 1, num_weights) * 32

        child1_binary = (parent1.binary_weights[:point1] + 
                        parent2.binary_weights[point1:point2] + 
                        parent1.binary_weights[point2:])
        child2_binary = (parent2.binary_weights[:point1] + 
                        parent1.binary_weights[point1:point2] + 
                        parent2.binary_weights[point2:])

        child1 = Individual(RobotNetwork())
        child2 = Individual(RobotNetwork())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_network_weights()
        child2.update_network_weights()

        return child1, child2

    def uniform_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32
        child1_binary = []
        child2_binary = []

        for i in range(num_weights):
            start_idx = i * 32
            end_idx = start_idx + 32
            # Decide for each 32-bit group which parent to take from
            if np.random.random() < 0.5:
                child1_binary.extend(parent1.binary_weights[start_idx:end_idx])
                child2_binary.extend(parent2.binary_weights[start_idx:end_idx])
            else:
                child1_binary.extend(parent2.binary_weights[start_idx:end_idx])
                child2_binary.extend(parent1.binary_weights[start_idx:end_idx])

        child1 = Individual(RobotNetwork())
        child2 = Individual(RobotNetwork())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_network_weights()
        child2.update_network_weights()

        return child1, child2

    def crossover_binary(self, parent1, parent2):
        # Podemos elegir aleatoriamente el tipo de cruce o establecerlo como parámetro
        crossover_type = np.random.choice(['one_point', 'two_point', 'uniform'])
        
        if crossover_type == 'one_point':
            # Original one-point crossover
            num_weights = len(parent1.binary_weights) // 32
            crossover_point = np.random.randint(0, num_weights) * 32
            child1_binary = parent1.binary_weights[:crossover_point] + parent2.binary_weights[crossover_point:]
            child2_binary = parent2.binary_weights[:crossover_point] + parent1.binary_weights[crossover_point:]

            child1 = Individual(RobotNetwork())
            child2 = Individual(RobotNetwork())

            child1.binary_weights = child1_binary
            child2.binary_weights = child2_binary

            child1.update_network_weights()
            child2.update_network_weights()

            return child1, child2
        elif crossover_type == 'two_point':
            return self.two_point_crossover(parent1, parent2)
        else:  # uniform
            return self.uniform_crossover(parent1, parent2)
    
    def mutate_real_valued(self, individual):
        std = (1 - (-1)) / 6
        
        # FC1 weights mutation
        mutation_mask_fc1_weight = torch.rand(individual.network.fc1.weight.shape) < self.mutation_rate
        mutation_values_fc1_weight = torch.randn(individual.network.fc1.weight.shape) * std
        individual.network.fc1.weight.data += mutation_mask_fc1_weight * mutation_values_fc1_weight
        
        # FC1 bias mutation
        mutation_mask_fc1_bias = torch.rand(individual.network.fc1.bias.shape) < self.mutation_rate
        mutation_values_fc1_bias = torch.randn(individual.network.fc1.bias.shape) * std
        individual.network.fc1.bias.data += mutation_mask_fc1_bias * mutation_values_fc1_bias
        
        # FC2 weights mutation
        mutation_mask_fc2_weight = torch.rand(individual.network.fc2.weight.shape) < self.mutation_rate
        mutation_values_fc2_weight = torch.randn(individual.network.fc2.weight.shape) * std
        individual.network.fc2.weight.data += mutation_mask_fc2_weight * mutation_values_fc2_weight
        
        # FC2 bias mutation
        mutation_mask_fc2_bias = torch.rand(individual.network.fc2.bias.shape) < self.mutation_rate
        mutation_values_fc2_bias = torch.randn(individual.network.fc2.bias.shape) * std
        individual.network.fc2.bias.data += mutation_mask_fc2_bias * mutation_values_fc2_bias
    
    def mutate_binary(self, individual):
        num_weights = len(individual.binary_weights) // 32
        for i in range(num_weights):
            if np.random.random() < self.mutation_rate:
                # Get the 32-bit group
                start_idx = i * 32
                end_idx = start_idx + 32
                weight_bits = individual.binary_weights[start_idx:end_idx]
                
                # Generate a new random float and convert it to binary
                new_weight = np.random.uniform(-1, 1)
                new_binary = format(struct.unpack('!I', struct.pack('!f', new_weight))[0], '032b')
                
                # Replace the old 32-bit group with the new one
                individual.binary_weights[start_idx:end_idx] = list(new_binary)
        
        individual.update_network_weights()