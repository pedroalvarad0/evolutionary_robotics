import numpy as np
from robot_network import SimpleRobotNetwork, RobotNetwork
from copy import deepcopy
import struct

def generate_random_weights_simple(input_size=9, output_size=2):
    network = SimpleRobotNetwork(input_size, output_size)

    weights_list = []

    for param in network.parameters():
        weights = param.data.numpy().flatten().tolist()
        weights_list.extend(weights)

    return weights_list


def generate_random_weights(input_size=9, hidden_size=10, output_size=2):
    network = RobotNetwork(input_size, hidden_size, output_size)

    weights_list = []

    for param in network.parameters():
        weights = param.data.numpy().flatten().tolist()
        weights_list.extend(weights)

    return weights_list


def move_object_fitness(object_positions_history):
    total_distance = 0
    for i in range(1, len(object_positions_history)):
        x_current, y_current = object_positions_history[i][0], object_positions_history[i][1]
        x_prev, y_prev = object_positions_history[i-1][0], object_positions_history[i-1][1]
        
        distance = np.sqrt((x_current - x_prev)**2 + (y_current - y_prev)**2)
        total_distance += distance
     
    return 1000 * total_distance

def fitness(object_positions_history, inputs_epuck1, inputs_epuck2):
    object_distance = move_object_fitness(object_positions_history)

    inputs_epuck1_data = np.array(inputs_epuck1)
    inputs_epuck2_data = np.array(inputs_epuck2)

    epuck1_light_sensor_data = inputs_epuck1_data[:, :8]
    epuck1_seeing_object_data = inputs_epuck1_data[:, 8]

    epuck2_light_sensor_data = inputs_epuck2_data[:, :8]
    epuck2_seeing_object_data = inputs_epuck2_data[:, 8]

    epuck1_steps_seeing_object = np.sum(epuck1_seeing_object_data)
    epuck2_steps_seeing_object = np.sum(epuck2_seeing_object_data)

    epuck1_light_perceived = np.sum(epuck1_light_sensor_data)
    epuck2_light_perceived = np.sum(epuck2_light_sensor_data)

    # print(f"epuck1_steps_seeing_object: {epuck1_steps_seeing_object}")
    # print(f"epuck2_steps_seeing_object: {epuck2_steps_seeing_object}")

    # print(f"epuck1_light_perceived: {epuck1_light_perceived}")
    # print(f"epuck2_light_perceived: {epuck2_light_perceived}")

    # print(f"object_distance: {1000 * object_distance}")

    fitness_ = 1000 * object_distance + 10 * (epuck1_steps_seeing_object + epuck2_steps_seeing_object) + (epuck1_light_perceived + epuck2_light_perceived) / 30

    return fitness_

class Individual:
    def __init__(self, weights):
        self.fitness = 0.0

        self.weights = weights
        self.binary_weights = self.weights_to_binary(self.weights)

    def weights_to_binary(self, weights):
        binary_representation = []
        for weight in weights:
            binary = format(struct.unpack('!I', struct.pack('!f', weight))[0], '032b')
            binary_representation.extend(list(binary))
        return binary_representation

    def binary_to_weights(self, binary_weights):
        weights = []
        for i in range(0, len(binary_weights), 32):
            binary_weight = ''.join(binary_weights[i:i+32])
            float_weight = struct.unpack('!f', struct.pack('!I', int(binary_weight, 2)))[0]
            weights.append(float_weight)
        return weights
    
    def update_weights(self):
        self.weights = self.binary_to_weights(self.binary_weights)

    def get_weights(self):
        return self.weights

    # def get_weights(self):
    #     midpoint = len(self.weights) // 2
    #     weights_network1 = self.weights[:midpoint]
    #     weights_network2 = self.weights[midpoint:]
        
    #     return weights_network1, weights_network2


class GeneticAlgorithm:
    def __init__(self, population_size=100, crossover_rate=0.8, mutation_rate=0.02, representation="binary"):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.representation = representation

    def generate_random_individual(self):
        return Individual(weights=generate_random_weights())
    
    def generate_initial_population(self):
        return [self.generate_random_individual() for _ in range(self.population_size)]

    def create_next_generation(self, population):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        fittest_individual = population[0]

        new_population = []

        while len(new_population) < len(population):
            parent1 = deepcopy(self.tournament_selection(population))
            parent2 = deepcopy(self.tournament_selection(population))

            parent1.fitness = 0
            parent2.fitness = 0

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
                child = deepcopy(np.random.choice([parent1, parent2]))
                if self.representation == "real":
                    self.mutate_real_valued(child)
                elif self.representation == "binary":
                    self.mutate_binary(child)

                new_population.append(child)

        return fittest_individual, new_population
    
    def tournament_selection(self, population, tournament_size=3):
        tournament = list(np.random.choice(population, tournament_size))
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def crossover_real_valued(self, parent1, parent2):
        beta = np.random.uniform(low=-0.25, high=1.25, size=len(parent1.weights))

        child_weights = (np.array(parent1.weights) * beta + np.array(parent2.weights) * (1 - beta)).tolist()

        child = Individual(weights=child_weights)

        return child
    
    def two_point_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32
        # Ensure points are aligned with 32-bit boundaries and point1 < point2
        point1 = np.random.randint(0, num_weights - 1) * 32
        point2 = np.random.randint(point1//32 + 1, num_weights) * 32

        child1_binary = parent1.binary_weights[:point1] + parent2.binary_weights[point1:point2] + parent1.binary_weights[point2:]
        child2_binary = parent2.binary_weights[:point1] + parent1.binary_weights[point1:point2] + parent2.binary_weights[point2:]

        child1 = Individual(weights=generate_random_weights())
        child2 = Individual(weights=generate_random_weights())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_weights()
        child2.update_weights()

        return child1, child2
    
    def one_point_crossover(self, parent1, parent2):
        num_weights = len(parent1.binary_weights) // 32
        crossover_point = np.random.randint(0, num_weights) * 32
        
        child1_binary = parent1.binary_weights[:crossover_point] + parent2.binary_weights[crossover_point:]
        child2_binary = parent2.binary_weights[:crossover_point] + parent1.binary_weights[crossover_point:]

        child1 = Individual(weights=generate_random_weights_simple())
        child2 = Individual(weights=generate_random_weights_simple())

        child1.binary_weights = child1_binary
        child2.binary_weights = child2_binary

        child1.update_weights()
        child2.update_weights()

        return child1, child2

    def crossover_binary(self, parent1, parent2):
        crossover_type = np.random.choice(['one_point', 'two_point'])
        
        if crossover_type == 'one_point':
            # Original one-point crossover
            return self.one_point_crossover(parent1, parent2)
        elif crossover_type == 'two_point':
            return self.two_point_crossover(parent1, parent2)
    
    def mutate_real_valued(self, individual):
        std = (1 - (-1)) / 6

        # Use numpy instead of torch for mutation operations
        mutation_mask = np.random.random(len(individual.weights)) < self.mutation_rate
        mutation_values = np.random.normal(0, std, len(individual.weights))

        weights_array = np.array(individual.weights)
        # Apply mutation using numpy operations
        weights_array = weights_array + (mutation_mask * mutation_values)
        individual.weights = weights_array.tolist()
    
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

        individual.update_weights()