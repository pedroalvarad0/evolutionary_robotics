"""explorer_controller.py"""

from controller import Supervisor
from robot_network import RobotNetwork
from genetic_algorithm import GeneticAlgorithm
import enum
import torch
import numpy as np
import uuid
import json

def load_robot_weights(robot_network, weights):
    idx = 0
    for param in robot_network.parameters():
        layer_size = param.data.numel()
        layer_weights = weights[idx:idx + layer_size]
        param.data = torch.tensor(layer_weights).reshape(param.data.shape)
        idx += layer_size


def get_np_image_from_camera(camera):
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    image_rgb = image_array[:, :, :3][:, :, ::-1]  # Convertir BGRA a RGB
    return image_rgb


def calculate_average_color(image):
    avg_color = np.mean(image, axis=(0,1))
    avg_color = np.round(avg_color).astype(int) / 255
    return avg_color


def save_data(initial_position, initial_rotation, ga_history):
    data_to_save = {
        "world_info": {
            "initial_position": initial_position,
            "initial_rotation": initial_rotation,
        },
        "individuals": []
    }

    for generation, individual in enumerate(ga_history):
        individual_data = {
            "generation": generation,
            "fitness": individual.fitness,
            "weights": individual.weights
        }
        data_to_save["individuals"].append(individual_data)
    
    filename = f"explorer_controller_history_{uuid.uuid4()}.json"
    with open(filename, "w") as f:
        json.dump(data_to_save, f)


def read_json_to_dict(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        return None


class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    EXECUTION = 2   # Mode for using the trained controller

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
MAX_TIME = 60

explorer_node = robot.getFromDef("EXPLORER")

translation_field = explorer_node.getField("translation")
rotation_field = explorer_node.getField("rotation")

INITIAL_POSITION = translation_field.getSFVec3f()
INITIAL_ROTATION = rotation_field.getSFRotation()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

ground_light_node = robot.getFromDef("GROUND_LIGHT")
ground_light_on_field = ground_light_node.getField("on")

ground_camera = robot.getDevice("ground_camera")
ground_camera.enable(timestep)

mode = Mode.EXECUTION

if mode == Mode.TRAINING:
    POPULATION_SIZE = 100
    GENERATIONS = 25

    current_individual = 0
    current_generation = 0
    current_time = 0
    previous_time = 0

    robot_network = RobotNetwork()

    ga_history = []

    genetic_algorithm = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        crossover_rate=0.8,
        mutation_rate=0.05,
        representation="binary"
    )

    population = genetic_algorithm.generate_initial_population()

    weights_network = population[0].get_weights()
    load_robot_weights(robot_network, weights_network)

    avg_color_history = []
    light_on_history = []

    # def fitness(avg_color_history, light_on_history):
    #     fitness = 0
    #     for i in range(len(avg_color_history)):
    #         fitness += avg_color_history[i] * light_on_history[i]
    #     return fitness

    def fitness(avg_color_history, light_on_history):
        color_array = np.array(avg_color_history)
        light_on_array = np.array(light_on_history)

        # Determinar cuando el color rojo es dominante
        red_dominant = np.logical_and(
            color_array[:, 0] > color_array[:, 1],
            color_array[:, 0] > color_array[:, 2]
        ).astype(int)
        
        # Invertir para saber cuando NO está en región roja
        not_red_dominant = 1 - red_dominant
        
        # Calcular componentes del fitness
        red_dominant_count = np.sum(red_dominant)  # Premio por estar en región roja
        light_on_with_red_dominant = np.sum(red_dominant * light_on_array)  # Premio por encender luz en región roja
        light_on_without_red = np.sum(not_red_dominant * light_on_array)  # Penalización por encender luz fuera de región roja
        
        # Fitness final: premios menos penalización
        return int(red_dominant_count + light_on_with_red_dominant - light_on_without_red)
        

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo
            population[current_individual].fitness = fitness(avg_color_history, light_on_history)
            
            current_individual += 1

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            ground_light_on_field.setSFBool(False)
            translation_field.setSFVec3f(INITIAL_POSITION)
            rotation_field.setSFRotation(INITIAL_ROTATION)
            explorer_node.resetPhysics()
            ground_light_on_field.setSFBool(False)

            if current_individual < POPULATION_SIZE:
                weights_network = population[current_individual].get_weights()
                load_robot_weights(robot_network, weights_network)

            avg_color_history = []
            light_on_history = []

            robot.step(timestep)

            if current_individual == POPULATION_SIZE:
                current_individual = 0
                fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")

                ga_history.append(fittest_individual)
                population = new_population
                current_generation += 1

                weights_network = population[0].get_weights()
                load_robot_weights(robot_network, weights_network)

                robot.step(timestep)
                if current_generation == GENERATIONS:
                    save_data(INITIAL_POSITION, INITIAL_ROTATION, ga_history)

        image_rgb = get_np_image_from_camera(ground_camera)
        avg_color = calculate_average_color(image_rgb)
        avg_color_history.append(avg_color)

        #print(avg_color)

        normalized_color = torch.tensor(avg_color).float()
        outputs = robot_network.forward(normalized_color)

        percentage_left_speed = outputs[0].item()
        percentage_right_speed = outputs[1].item()
        ground_light_on_prob = outputs[2].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)

        if ground_light_on_prob > 0.5:
            light_on_history.append(1)
            ground_light_on_field.setSFBool(True)
        else:
            light_on_history.append(0)
            ground_light_on_field.setSFBool(False)
elif mode == Mode.EXECUTION:
    file_path = "explorer_controller_history_619fb75c-4c95-4f29-978e-b4c9c7b7d21a.json"
    data_dict = read_json_to_dict(file_path)

    robot_network = RobotNetwork()
    weights_network = data_dict["individuals"][-1]["weights"]
    load_robot_weights(robot_network, weights_network)

    translation_field.setSFVec3f(data_dict["world_info"]["initial_position"])
    rotation_field.setSFRotation(data_dict["world_info"]["initial_rotation"])

    while robot.step(timestep) != -1:
        image_rgb = get_np_image_from_camera(ground_camera)
        avg_color = calculate_average_color(image_rgb)
        #avg_color_history.append(avg_color)

        normalized_color = torch.tensor(avg_color).float()
        outputs = robot_network.forward(normalized_color)

        percentage_left_speed = outputs[0].item()
        percentage_right_speed = outputs[1].item()
        ground_light_on_prob = outputs[2].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)

        if ground_light_on_prob > 0.5:
            ground_light_on_field.setSFBool(True)
        else:
            ground_light_on_field.setSFBool(False)
    