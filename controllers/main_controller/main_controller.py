"""main_controller controller."""
from controller import Robot, Supervisor
from genetic_algorithm import GeneticAlgorithm, fitness
import numpy as np
import torch
import json
from utils import get_sensor_values, normalize_sensor_values, load_robot_weights, create_config_file, save_generation_data, read_json_to_dict
from robot_network import RobotNetwork
import uuid
import enum


class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    CONTINUE_TRAINING = 2   # Mode for continuing the training
    EXECUTION = 3   # Mode for using the trained controller

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
MAX_TIME = 60
POPULATION_SIZE = 50
GENERATIONS = 500
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.02
REPRESENTATION = "binary"

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# se realiza para ambos robots sin importar el nombre
distance_sensors = []
light_sensors = []

distance_sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]

for i in range(len(distance_sensor_names)):
    sensor = robot.getDevice(distance_sensor_names[i])
    sensor.enable(timestep)
    distance_sensors.append(sensor)

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

mode = Mode.TRAINING

if mode == Mode.TRAINING:
    if robot_name == "robot1":
        ga_uuid = uuid.uuid4()


        # obtener nodos de los robots
        robot1_node = robot.getFromDef("ROBOT1")
        robot2_node = robot.getFromDef("ROBOT2")

        box_node = robot.getFromDef("BOX")
        area_node = robot.getFromDef("AREA")

        robot1_ground_light_node = robot.getFromDef("GROUND_LIGHT_ROBOT1")
        robot1_ground_light_on_field = robot1_ground_light_node.getField("on")

        translation_field_robot1 = robot1_node.getField("translation")
        rotation_field_robot1 = robot1_node.getField("rotation")

        translation_field_robot2 = robot2_node.getField("translation")
        rotation_field_robot2 = robot2_node.getField("rotation")

        translation_field_box = box_node.getField("translation")
        rotation_field_box = box_node.getField("rotation")

        translation_field_area = area_node.getField("translation")

        custom_data_field_robot1 = robot1_node.getField("customData")
        custom_data_field_robot2 = robot2_node.getField("customData")

        INITIAL_POSITION_ROBOT1 = translation_field_robot1.getSFVec3f()
        INITIAL_ROTATION_ROBOT1 = rotation_field_robot1.getSFRotation()

        INITIAL_POSITION_ROBOT2 = translation_field_robot2.getSFVec3f()
        INITIAL_ROTATION_ROBOT2 = rotation_field_robot2.getSFRotation()

        INITIAL_POSITION_BOX = translation_field_box.getSFVec3f()
        INITIAL_ROTATION_BOX = rotation_field_box.getSFRotation()

        INITIAL_POSITION_AREA = translation_field_area.getSFVec3f()
        
        create_config_file(
            ga_uuid,
            MAX_TIME,
            POPULATION_SIZE,
            GENERATIONS,
            CROSSOVER_RATE,
            MUTATION_RATE,
            REPRESENTATION,
            INITIAL_POSITION_ROBOT1,
            INITIAL_ROTATION_ROBOT1,
            INITIAL_POSITION_ROBOT2,
            INITIAL_ROTATION_ROBOT2,
            INITIAL_POSITION_BOX,
            INITIAL_ROTATION_BOX,
            INITIAL_POSITION_AREA
        )

        # variables para el algoritmo genetico
        current_individual = 0
        current_generation = 0
        current_time = 0
        previous_time = 0

        robot_network = RobotNetwork()

        #ga_history = []
        genetic_algorithm = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            representation=REPRESENTATION
        )
        population = genetic_algorithm.generate_initial_population()

        weights_network1, weights_network2 = population[0].get_weights()

        # cargar pesos iniciales
        load_robot_weights(robot_network, weights_network1)
        
        # enviar pesos iniciales a main2
        custom_data_field_robot2.setSFString(str(weights_network2))
        robot.step(timestep)

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time: # nuevo individuo
                population[current_individual].fitness = fitness(translation_field_box.getSFVec3f(), INITIAL_POSITION_AREA)

                current_individual += 1
                
                # resetear valores
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                
                robot1_ground_light_on_field.setSFBool(False)

                robot1_node.resetPhysics()
                robot2_node.resetPhysics()

                box_node.resetPhysics()
                
                translation_field_robot1.setSFVec3f(INITIAL_POSITION_ROBOT1)
                rotation_field_robot1.setSFRotation(INITIAL_ROTATION_ROBOT1)
                
                translation_field_robot2.setSFVec3f(INITIAL_POSITION_ROBOT2)
                rotation_field_robot2.setSFRotation(INITIAL_ROTATION_ROBOT2)
                
                translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
                rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

                #translation_field_area.setSFVec3f(INITIAL_POSITION_AREA)

                if current_individual < POPULATION_SIZE:
                    weights_network1, weights_network2 = population[current_individual].get_weights()
                    load_robot_weights(robot_network, weights_network1)
                    custom_data_field_robot2.setSFString(str(weights_network2))

                robot.step(timestep)

                if current_individual == POPULATION_SIZE:
                    current_individual = 0
                    fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                    print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")
                    save_generation_data(fittest_individual, population, current_generation, ga_uuid)

                    #ga_history.append(fittest_individual)
                    population = new_population
                    current_generation += 1

                    # send nuevos pesos a main2
                    weights_network1, weights_network2 = population[0].get_weights()
                    custom_data_field_robot2.setSFString(str(weights_network2))
                    load_robot_weights(robot_network, weights_network1)

                    robot.step(timestep)
                    if current_generation == GENERATIONS:
                        pass

            # obtener valores de los sensores
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            
            distance_sensor_values = get_sensor_values(distance_sensors)
            normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))

            # obtener direcciones
            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            turn_light_on_prob = outputs[2].item()

            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

            if turn_light_on_prob > 0.5:
                robot1_ground_light_on_field.setSFBool(True)
            else:
                robot1_ground_light_on_field.setSFBool(False)

    elif robot_name == "robot2":
        robot.step(timestep)

        robot2_node = robot.getFromDef("ROBOT2")
        robot2_ground_light_node = robot.getFromDef("GROUND_LIGHT_ROBOT2")
        robot2_ground_light_on_field = robot2_ground_light_node.getField("on")

        custom_data_field_robot2 = robot2_node.getField("customData")
        weights = json.loads(custom_data_field_robot2.getSFString())
        robot_network = RobotNetwork()
        load_robot_weights(robot_network, weights)

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time: # nuevo individuo
                robot2_ground_light_on_field.setSFBool(False)

                # obtener nuevos pesos para main2
                weights = json.loads(custom_data_field_robot2.getSFString())
                load_robot_weights(robot_network, weights)
            
            # obtener valores de los sensores
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            
            distance_sensor_values = get_sensor_values(distance_sensors)
            normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))
            
            # obtener direcciones
            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            turn_light_on_prob = outputs[2].item()

            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

            if turn_light_on_prob > 0.5:
                robot2_ground_light_on_field.setSFBool(True)
            else:
                robot2_ground_light_on_field.setSFBool(False)
elif mode == Mode.CONTINUE_TRAINING:
    pass
elif mode == Mode.EXECUTION:
    if robot_name == "robot1":
        ga_uuid_string = "1775544e-eb6f-4e86-aa0a-846d56a77486"
        generation_file = "generation_127.json"

        robot1_ground_light_node = robot.getFromDef("GROUND_LIGHT_ROBOT1")
        robot1_ground_light_on_field = robot1_ground_light_node.getField("on")

        custom_data_field_robot2 = robot.getFromDef("ROBOT2").getField("customData")

        generation_data = read_json_to_dict(f"histories/{ga_uuid_string}/{generation_file}")

        weights = generation_data["fittest_individual_weights"]

        midpoint = len(weights) // 2
        weights_network1 = weights[:midpoint]
        weights_network2 = weights[midpoint:]

        robot_network = RobotNetwork()
        load_robot_weights(robot_network, weights_network1)

        custom_data_field_robot2.setSFString(str(weights_network2))

        robot.step(timestep)

        while robot.step(timestep) != -1:
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            
            distance_sensor_values = get_sensor_values(distance_sensors)
            normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))

            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            turn_light_on_prob = outputs[2].item()
            
            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

            if turn_light_on_prob > 0.5:
                robot1_ground_light_on_field.setSFBool(True)
            else:
                robot1_ground_light_on_field.setSFBool(False)

    elif robot_name == "robot2":
        robot.step(timestep)

        robot2_node = robot.getFromDef("ROBOT2")
        robot2_ground_light_node = robot.getFromDef("GROUND_LIGHT_ROBOT2")
        robot2_ground_light_on_field = robot2_ground_light_node.getField("on")

        custom_data_field_robot2 = robot2_node.getField("customData")
        weights = json.loads(custom_data_field_robot2.getSFString())

        robot_network = RobotNetwork()
        load_robot_weights(robot_network, weights)

        while robot.step(timestep) != -1:
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)

            distance_sensor_values = get_sensor_values(distance_sensors)
            normalized_distance_sensor_values = normalize_sensor_values(distance_sensor_values, 0, 1000)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(normalized_distance_sensor_values)))

            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            turn_light_on_prob = outputs[2].item()
            
            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

            if turn_light_on_prob > 0.5:
                robot2_ground_light_on_field.setSFBool(True)
            else:
                robot2_ground_light_on_field.setSFBool(False)
# main loop
