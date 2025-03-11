"""main_controller controller."""
from controller import Robot, Supervisor
from genetic_algorithm import GeneticAlgorithm
import numpy as np
import torch
import json
from utils import get_sensor_values, normalize_sensor_values, get_np_image_from_camera, calculate_average_color
from robot_network import RobotNetwork
import uuid

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
MAX_TIME = 30
POPULATION_SIZE = 100
GENERATIONS = 100

def load_robot_weights(robot_network, weights):
    idx = 0
    for param in robot_network.parameters():
        layer_size = param.data.numel()
        layer_weights = weights[idx:idx + layer_size]
        param.data = torch.tensor(layer_weights).reshape(param.data.shape)
        idx += layer_size

def save_data(ga_history):
    data_to_save = []
    for generation, individual in enumerate(ga_history):
        individual_data = {
            "generation": generation,
            "fitness": individual.fitness,
            "weights_robot1": individual.weights_robot1,
            "weights_robot2": individual.weights_robot2
        }
        data_to_save.append(individual_data)
    
    filename = f"ga_history_{uuid.uuid4()}.json"
    with open(filename, "w") as f:
        json.dump(data_to_save, f)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")
camera.enable(timestep)

# se realiza para ambos robots sin importar el nombre
distance_sensors = []
light_sensors = []

distance_sensor_names = ["ps0", "ps1", "ps6", "ps7"]
light_sensor_names = ["ls0", "ls1", "ls6", "ls7"]

for i in range(len(distance_sensor_names)):
    sensor = robot.getDevice(distance_sensor_names[i])
    sensor.enable(timestep)
    distance_sensors.append(sensor)

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

# ------------------------------------------------------------
# fitness
def fitness(avg_camera_color_history, light_history):
    color_array_main1 = np.array(avg_camera_color_history["main1"])
    color_array_main2 = np.array(avg_camera_color_history["main2"])

    light_array_main1 = np.array(light_history["main1"])
    light_array_main2 = np.array(light_history["main2"])
    
    red_dominant_main1 = np.logical_and(
        color_array_main1[:, 0] > color_array_main1[:, 1],
        color_array_main1[:, 0] > color_array_main1[:, 2]
    ).astype(int)
    
    red_dominant_main2 = np.logical_and(
        color_array_main2[:, 0] > color_array_main2[:, 1],
        color_array_main2[:, 0] > color_array_main2[:, 2]
    ).astype(int)
    
    red_dominant_count_main1 = np.sum(red_dominant_main1)
    red_dominant_count_main2 = np.sum(red_dominant_main2)
    
    complement_light_array_main1 = 1 - light_array_main1
    complement_light_array_main2 = 1 - light_array_main2
    
    light_array_sum_main1 = np.sum(red_dominant_main1 * np.sum(complement_light_array_main1, axis=1))
    light_array_sum_main2 = np.sum(red_dominant_main2 * np.sum(complement_light_array_main2, axis=1))
    
    fitness_main1 = red_dominant_count_main1 + light_array_sum_main1
    fitness_main2 = red_dominant_count_main2 + light_array_sum_main2
    
    return fitness_main1, fitness_main2
    

# main loop
if robot_name == "main1":
    # obtener nodos de los robots
    robot_main1_node = robot.getFromDef("MAIN1")
    robot_main2_node = robot.getFromDef("MAIN2")

    box_node = robot.getFromDef("BOX")
    light_node = robot.getFromDef("LIGHT")

    translation_field_main1 = robot_main1_node.getField("translation")
    rotation_field_main1 = robot_main1_node.getField("rotation")

    translation_field_main2 = robot_main2_node.getField("translation")
    rotation_field_main2 = robot_main2_node.getField("rotation")

    translation_field_box = box_node.getField("translation")
    rotation_field_box = box_node.getField("rotation")

    translation_field_light = light_node.getField("translation")

    custom_data_field_main1 = robot_main1_node.getField("customData")
    custom_data_field_main2 = robot_main2_node.getField("customData")

    INITIAL_POSITION_MAIN1 = translation_field_main1.getSFVec3f()
    INITIAL_ROTATION_MAIN1 = rotation_field_main1.getSFRotation()

    INITIAL_POSITION_MAIN2 = translation_field_main2.getSFVec3f()
    INITIAL_ROTATION_MAIN2 = rotation_field_main2.getSFRotation()

    INITIAL_POSITION_BOX = translation_field_box.getSFVec3f()
    INITIAL_ROTATION_BOX = rotation_field_box.getSFRotation()

    LIGHT_POSITION = translation_field_light.getSFVec3f()

    # obtener receiver
    main1_receiver = robot.getDevice("receiver")
    main1_receiver.enable(timestep)

    # variables para el algoritmo genetico
    current_individual = 0
    current_generation = 0
    current_time = 0
    previous_time = 0

    robot_network = RobotNetwork()

    ga_history = []
    genetic_algorithm = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.05,
        representation="real"
    )
    population = genetic_algorithm.generate_initial_population()

    weights_network1, weights_network2 = population[0].get_weights()

    # cargar pesos iniciales
    load_robot_weights(robot_network, weights_network1)
    
    # enviar pesos iniciales a main2
    custom_data_field_main2.setSFString(str(weights_network2))
    robot.step(timestep)

    current_time = 0
    previous_time = 0

    # historial de valores de los sensores
    light_history = {
        "main1": [],
        "main2": []
    }

    avg_camera_color_history = {
        "main1": [],
        "main2": []
    }

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo

            fitness_main1, fitness_main2 = fitness(avg_camera_color_history, light_history)

            # media armonica
            #population[current_individual].fitness = (2 * fitness_main1 * fitness_main2) / (fitness_main1 + fitness_main2 + 0.01)

            # Media ponderada con penalización de desbalance
            population[current_individual].fitness = ((fitness_main1 + fitness_main2) / 2) - (0.5) * abs(fitness_main1 - fitness_main2)
            
            print(f"[{current_individual}]fitness_main1: {fitness_main1}, fitness_main2: {fitness_main2}, fitness: {population[current_individual].fitness}")

            current_individual += 1
            
            # resetear valores
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            
            robot.step(timestep)

            robot_main1_node.resetPhysics()
            robot_main2_node.resetPhysics()

            box_node.resetPhysics()
            
            translation_field_main1.setSFVec3f(INITIAL_POSITION_MAIN1)
            rotation_field_main1.setSFRotation(INITIAL_ROTATION_MAIN1)
            
            translation_field_main2.setSFVec3f(INITIAL_POSITION_MAIN2)
            rotation_field_main2.setSFRotation(INITIAL_ROTATION_MAIN2)
            
            translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
            rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

            if current_individual < POPULATION_SIZE:
                weights_network1, weights_network2 = population[current_individual].get_weights()
                load_robot_weights(robot_network, weights_network1)
                custom_data_field_main2.setSFString(str(weights_network2))
            
            light_history = {
                "main1": [],
                "main2": []
            }

            avg_camera_color_history = {
                "main1": [],
                "main2": []
            }

            robot.step(timestep)

            if current_individual == POPULATION_SIZE:
                current_individual = 0
                fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")

                ga_history.append(fittest_individual)
                population = new_population
                current_generation += 1

                # send nuevos pesos a main2
                weights_network1, weights_network2 = population[0].get_weights()
                custom_data_field_main2.setSFString(str(weights_network2))
                load_robot_weights(robot_network, weights_network1)


                robot.step(timestep)
                if current_generation == GENERATIONS:
                    save_data(ga_history)

        # obtener valores de los sensores
        image_rgb = get_np_image_from_camera(camera)
        avg_camera_color = torch.tensor(calculate_average_color(image_rgb)).float()

        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
        
        distance_sensor_values = get_sensor_values(distance_sensors)
        normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

        normalized_sensor_values = torch.cat((avg_camera_color, torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))

        # guardar valores de los sensores
        light_history["main1"].append(normalized_light_sensor_values)

        if main1_receiver.getQueueLength() > 0:
            main2_normalized_light_sensor_values = json.loads(main1_receiver.getString())
            light_history["main2"].append(main2_normalized_light_sensor_values)
            main1_receiver.nextPacket()

        avg_camera_color_history["main1"].append(avg_camera_color.tolist())

        if main1_receiver.getQueueLength() > 0:
            main2_avg_camera_color = json.loads(main1_receiver.getString())
            avg_camera_color_history["main2"].append(main2_avg_camera_color)
            main1_receiver.nextPacket()

        # obtener direcciones
        directions = robot_network.forward(normalized_sensor_values)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)

elif robot_name == "main2":

    main2_emitter = robot.getDevice("emitter")

    robot.step(timestep)

    robot_main2_node = robot.getFromDef("MAIN2")
    custom_data_field_main2 = robot_main2_node.getField("customData")
    weights = json.loads(custom_data_field_main2.getSFString())
    robot_network = RobotNetwork()
    load_robot_weights(robot_network, weights)

    current_time = 0
    previous_time = 0

    while robot.step(timestep) != -1:
        previous_time = current_time
        current_time = robot.getTime() % MAX_TIME

        if previous_time > current_time: # nuevo individuo

            # obtener nuevos pesos para main2
            weights = json.loads(custom_data_field_main2.getSFString())
            load_robot_weights(robot_network, weights)
        
        # obtener valores de los sensores
        image_rgb = get_np_image_from_camera(camera)
        avg_camera_color = torch.tensor(calculate_average_color(image_rgb)).float()

        light_sensor_values = get_sensor_values(light_sensors)
        normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
        
        distance_sensor_values = get_sensor_values(distance_sensors)
        normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

        normalized_sensor_values = torch.cat((avg_camera_color, torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))

        main2_emitter.send(str(normalized_light_sensor_values))
        main2_emitter.send(str(avg_camera_color.tolist()))
        
        # obtener direcciones
        directions = robot_network.forward(normalized_sensor_values)
        percentage_left_speed = directions[0].item()
        percentage_right_speed = directions[1].item()

        left_motor_velocity = percentage_left_speed * MAX_SPEED
        right_motor_velocity = percentage_right_speed * MAX_SPEED

        left_motor.setVelocity(left_motor_velocity)
        right_motor.setVelocity(right_motor_velocity)