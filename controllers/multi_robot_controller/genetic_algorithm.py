import torch
import struct
import numpy as np
from robot_network import RobotNetwork
#from torch.distributions.uniform import Uniform

class Individual:
    def __init__(self, robot1_network, robot2_network):
        self.fitness = 0.0
        self.robot1_network = robot1_network
        self.robot2_network = robot2_network
        self.binary_weights = self.weights_to_binary()

    def weights_to_binary(self):
        binary_representation = []

        for param in self.robot1_network.parameters():
            weights = param.data.numpy().flatten()
            for weight in weights:
                binary = format(struct.unpack('!I', struct.pack('!f', weight))[0], '032b')
                binary_representation.extend(list(binary))
                
        for param in self.robot2_network.parameters():
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

        weights_robot1 = weights[:len(weights)//2]
        weights_robot2 = weights[len(weights)//2:]

        idx = 0
        for param in self.robot1_network.parameters():
            layer_size = param.data.numel()
            layer_weights = weights_robot1[idx:idx + layer_size]
            param.data = torch.tensor(layer_weights).reshape(param.data.shape)
            idx += layer_size
            
        idx = 0
        for param in self.robot2_network.parameters():
            layer_size = param.data.numel()
            layer_weights = weights_robot2[idx:idx + layer_size]
            param.data = torch.tensor(layer_weights).reshape(param.data.shape)
            idx += layer_size
            

class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=100, crossover_rate=0.8, mutation_rate=0.02):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def generate_random_individual(self):
        return Individual(robot1_network=RobotNetwork(), robot2_network=RobotNetwork())
    
    def generate_initial_population(self):
        return [self.generate_random_individual() for _ in range(self.population_size)]
        
    def create_next_generation(self, population):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        fittest_individual = population[0]

        new_population = []
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            if np.random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1)
                self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(np.random.choice([parent1, parent2]))

        return fittest_individual, new_population
                
    def tournament_selection(self, population, tournament_size=10):
        tournament = list(np.random.choice(population, tournament_size))
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def one_point_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32

        crossover_point = np.random.randint(0, num_weights) * 32

        child1_binary = parent1.binary_weights[:crossover_point] + parent2.binary_weights[crossover_point:]
        child2_binary = parent2.binary_weights[:crossover_point] + parent1.binary_weights[crossover_point:]

        child1 = Individual(robot1_network=RobotNetwork(), robot2_network=RobotNetwork())
        child2 = Individual(robot1_network=RobotNetwork(), robot2_network=RobotNetwork())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_network_weights()
        child2.update_network_weights()

        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32

        crossover_point1 = np.random.randint(0, num_weights - 1) * 32
        crossover_point2 = np.random.randint(crossover_point1//32 + 1, num_weights) * 32

        child1_binary = parent1.binary_weights[:crossover_point1] + parent2.binary_weights[crossover_point1:crossover_point2] + parent1.binary_weights[crossover_point2:]
        child2_binary = parent2.binary_weights[:crossover_point1] + parent1.binary_weights[crossover_point1:crossover_point2] + parent2.binary_weights[crossover_point2:]
        
        child1 = Individual(robot1_network=RobotNetwork(), robot2_network=RobotNetwork())
        child2 = Individual(robot1_network=RobotNetwork(), robot2_network=RobotNetwork())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_network_weights()
        child2.update_network_weights()

        return child1, child2

    def crossover(self, parent1, parent2):
        crossover_type = np.random.choice(['one_point', 'two_point'])

        if crossover_type == 'one_point':
            child1, child2 = self.one_point_crossover(parent1, parent2)
        elif crossover_type == 'two_point':
            child1, child2 = self.two_point_crossover(parent1, parent2)

        return child1, child2
    
    def mutate(self, individual):
        for i in range(len(individual.binary_weights)):
            if np.random.random() < self.mutation_rate:
                individual.binary_weights[i] = '1' if individual.binary_weights[i] == '0' else '0'
        
        individual.update_network_weights()
                
    
    
            