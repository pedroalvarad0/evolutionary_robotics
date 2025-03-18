"""main_controller controller."""
from controller import Robot, Supervisor
from genetic_algorithm import GeneticAlgorithm, fitness, move_box_fitness
import numpy as np
import torch
import json
from utils import get_sensor_values,normalize_sensor_values, load_robot_weights, create_config_file, save_generation_data, read_json_to_dict, get_np_image_from_camera, calculate_average_color, get_history_info, get_last_generation_info
from robot_network import RobotNetwork, SimpleRobotNetwork
import uuid
import enum
import time

class Mode(enum.Enum):
    TRAINING = 1    # Mode for training the controller
    CONTINUE_TRAINING = 2   # Mode for continuing the training
    EXECUTION = 3   # Mode for using the trained controller
    REPLAY = 4
    TESTING = 5   # Mode for testing the controller

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep() / 2)

MAX_SPEED = 6.28
MAX_TIME = 60
POPULATION_SIZE = 25
GENERATIONS = 500
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
REPRESENTATION = "real"

camera = robot.getDevice("camera")
camera.enable(timestep)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

light_sensors = []
light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]

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

        #box_node = robot.getFromDef("BOX")
        object_node = robot.getFromDef("OBJECT")
        area_node = robot.getFromDef("AREA")

        translation_field_robot1 = robot1_node.getField("translation")
        rotation_field_robot1 = robot1_node.getField("rotation")

        translation_field_robot2 = robot2_node.getField("translation")
        rotation_field_robot2 = robot2_node.getField("rotation")

        translation_field_object = object_node.getField("translation")
        rotation_field_object = object_node.getField("rotation")

        translation_field_area = area_node.getField("translation")

        custom_data_field_robot1 = robot1_node.getField("customData")
        custom_data_field_robot2 = robot2_node.getField("customData")

        INITIAL_POSITION_ROBOT1 = translation_field_robot1.getSFVec3f()
        INITIAL_ROTATION_ROBOT1 = rotation_field_robot1.getSFRotation()

        INITIAL_POSITION_ROBOT2 = translation_field_robot2.getSFVec3f()
        INITIAL_ROTATION_ROBOT2 = rotation_field_robot2.getSFRotation()

        INITIAL_POSITION_OBJECT = translation_field_object.getSFVec3f()
        INITIAL_ROTATION_OBJECT = rotation_field_object.getSFRotation()

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
            INITIAL_POSITION_OBJECT,
            INITIAL_ROTATION_OBJECT,
        )

        # variables para el algoritmo genetico
        current_individual = 0
        current_generation = 0
        current_time = 0
        previous_time = 0

        robot_network = SimpleRobotNetwork()

        box_positions_history = []

        #ga_history = []
        genetic_algorithm = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            representation=REPRESENTATION
        )
        population = genetic_algorithm.generate_initial_population()

        weights_network = population[0].get_weights()

        # cargar pesos iniciales
        load_robot_weights(robot_network, weights_network)

        robot1_receiver = robot.getDevice("receiver")
        robot1_receiver.enable(timestep)

        robot1_emitter = robot.getDevice("emitter")

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time: # nuevo individuo
                population[current_individual].fitness = 10 * fitness(translation_field_object.getSFVec3f(), INITIAL_POSITION_AREA)
                #population[current_individual].fitness = move_box_fitness(box_positions_history)

                current_individual += 1
                
                # resetear valores
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                robot1_node.resetPhysics()
                robot2_node.resetPhysics()

                object_node.resetPhysics()
                
                translation_field_robot1.setSFVec3f(INITIAL_POSITION_ROBOT1)
                rotation_field_robot1.setSFRotation(INITIAL_ROTATION_ROBOT1)
                
                translation_field_robot2.setSFVec3f(INITIAL_POSITION_ROBOT2)
                rotation_field_robot2.setSFRotation(INITIAL_ROTATION_ROBOT2)
                
                #translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
                #rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

                translation_field_object.setSFVec3f(INITIAL_POSITION_OBJECT)
                rotation_field_object.setSFRotation(INITIAL_ROTATION_OBJECT)

                #translation_field_area.setSFVec3f(INITIAL_POSITION_AREA)

                if current_individual < POPULATION_SIZE:
                    weights = population[current_individual].get_weights()
                    load_robot_weights(robot_network, weights)
                    #custom_data_field_robot2.setSFString(str(weights_network2))

                box_positions_history = []

                #robot.step(timestep)

                if current_individual == POPULATION_SIZE:
                    current_individual = 0
                    fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                    print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")
                    save_generation_data(fittest_individual, population, current_generation, ga_uuid)

                    #ga_history.append(fittest_individual)
                    population = new_population
                    current_generation += 1

                    # send nuevos pesos a main2
                    weights = population[0].get_weights()
                    load_robot_weights(robot_network, weights)

                    # print(fittest_individual.weights)
                    # print(population[0].weights)

                    if current_generation == GENERATIONS:
                        pass
            
            box_positions_history.append(translation_field_object.getSFVec3f())

            robot2_inputs = torch.zeros(9) # 9 inputs: 8 light + 1 seeing object

            if robot1_receiver.getQueueLength() > 0:
                robot2_inputs = torch.tensor(json.loads(robot1_receiver.getString()))
                #print(f"Received at {time.time()}")
                robot1_receiver.nextPacket()

            # obtener valores de los sensores
            robot1_light_sensor_values = get_sensor_values(light_sensors)
            robot1_normalized_light_sensor_values = normalize_sensor_values(robot1_light_sensor_values, 0, 4095)
            robot1_discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in robot1_normalized_light_sensor_values]
            robot1_image = get_np_image_from_camera(camera)
            robot1_average_color = calculate_average_color(robot1_image)
            robot1_seeing_object = np.logical_and(robot1_average_color[0] > robot1_average_color[1], robot1_average_color[0] > robot1_average_color[2]).astype(int)
            robot1_inputs = torch.cat((torch.tensor(robot1_discretize_light_sensor_values).to(torch.float32), torch.tensor([robot1_seeing_object]).to(torch.float32)))
            
            # print(f"robot1_inputs: {robot1_inputs}")
            # print(f"robot2_inputs: {robot2_inputs}")

            inputs = torch.cat((robot1_inputs, robot2_inputs)).to(torch.float32)

            # obtener direcciones
            outputs = robot_network.forward(inputs)
            robot1_percentage_left_speed = outputs[0].item()
            robot1_percentage_right_speed = outputs[1].item()

            robot2_percentage_left_speed = outputs[2].item()
            robot2_percentage_right_speed = outputs[3].item()
            
            robot1_emitter.send(str([robot2_percentage_left_speed, robot2_percentage_right_speed]))
            #print(f"sent at {time.time()}")

            left_motor_velocity = robot1_percentage_left_speed * MAX_SPEED
            right_motor_velocity = robot1_percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

    elif robot_name == "robot2":
        #robot.step(timestep)

        robot2_node = robot.getFromDef("ROBOT2")

        robot2_emitter = robot.getDevice("emitter")

        robot2_receiver = robot.getDevice("receiver")
        robot2_receiver.enable(timestep)

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
            
            # obtener valores de los sensores
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image).tolist()
            seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)
            robot2_emitter.send(str(discretize_light_sensor_values + [seeing_object]))
            #print(f"Sent at {time.time()}")

            directions = [0, 0]
            if robot2_receiver.getQueueLength() > 0:
                directions = json.loads(robot2_receiver.getString())
                #print(f"Received at {time.time()}")
                robot2_receiver.nextPacket()

            left_motor_velocity = directions[0] * MAX_SPEED
            right_motor_velocity = directions[1] * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
elif mode == Mode.CONTINUE_TRAINING:
    if robot_name == "robot1":
        ga_uuid_string = "11cf44b2-e3e8-49e5-a806-ede387de9328"
        config, gens_info = get_history_info(ga_uuid_string)
        last_generation_info = get_last_generation_info(ga_uuid_string)

        print(f"Reanudando entrenamiento de {ga_uuid_string}. Generation: {last_generation_info['generation']}")

        robot1_node = robot.getFromDef("ROBOT1")
        robot2_node = robot.getFromDef("ROBOT2")

        object_node = robot.getFromDef("OBJECT")
        area_node = robot.getFromDef("AREA")
        
        translation_field_robot1 = robot1_node.getField("translation")
        rotation_field_robot1 = robot1_node.getField("rotation")

        translation_field_robot2 = robot2_node.getField("translation")
        rotation_field_robot2 = robot2_node.getField("rotation")

        translation_field_object = object_node.getField("translation")
        rotation_field_object = object_node.getField("rotation")

        translation_field_area = area_node.getField("translation")

        custom_data_field_robot1 = robot1_node.getField("customData")
        custom_data_field_robot2 = robot2_node.getField("customData")

        INITIAL_POSITION_ROBOT1 = translation_field_robot1.getSFVec3f()
        INITIAL_ROTATION_ROBOT1 = rotation_field_robot1.getSFRotation()

        INITIAL_POSITION_ROBOT2 = translation_field_robot2.getSFVec3f()
        INITIAL_ROTATION_ROBOT2 = rotation_field_robot2.getSFRotation()

        INITIAL_POSITION_OBJECT = translation_field_object.getSFVec3f()
        INITIAL_ROTATION_OBJECT = rotation_field_object.getSFRotation()

        INITIAL_POSITION_AREA = translation_field_area.getSFVec3f()

        current_individual = 0
        current_generation = last_generation_info["generation"]

        robot_network = SimpleRobotNetwork()

        genetic_algorithm = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            representation=REPRESENTATION
        )
        population = genetic_algorithm.generate_initial_population()

        # load weights
        for i in range(len(population)):
            population[i].weights = last_generation_info["population"][i]["weights"]
            population[i].update_weights()

        weights = population[current_individual].get_weights()
        load_robot_weights(robot_network, weights)

        robot1_receiver = robot.getDevice("receiver")
        robot1_receiver.enable(timestep)
        
        robot1_emitter = robot.getDevice("emitter")

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time: # nuevo individuo
                population[current_individual].fitness = 10 * fitness(translation_field_object.getSFVec3f(), INITIAL_POSITION_AREA)
                #population[current_individual].fitness = move_box_fitness(box_positions_history)

                current_individual += 1
                
                # resetear valores
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                robot1_node.resetPhysics()
                robot2_node.resetPhysics()

                object_node.resetPhysics()
                
                translation_field_robot1.setSFVec3f(INITIAL_POSITION_ROBOT1)
                rotation_field_robot1.setSFRotation(INITIAL_ROTATION_ROBOT1)
                
                translation_field_robot2.setSFVec3f(INITIAL_POSITION_ROBOT2)
                rotation_field_robot2.setSFRotation(INITIAL_ROTATION_ROBOT2)
                
                #translation_field_box.setSFVec3f(INITIAL_POSITION_BOX)
                #rotation_field_box.setSFRotation(INITIAL_ROTATION_BOX)

                translation_field_object.setSFVec3f(INITIAL_POSITION_OBJECT)
                rotation_field_object.setSFRotation(INITIAL_ROTATION_OBJECT)

                #translation_field_area.setSFVec3f(INITIAL_POSITION_AREA)

                if current_individual < POPULATION_SIZE:
                    weights = population[current_individual].get_weights()
                    load_robot_weights(robot_network, weights)
                    #custom_data_field_robot2.setSFString(str(weights_network2))

                robot.step(timestep)

                if current_individual == POPULATION_SIZE:
                    current_individual = 0
                    fittest_individual, new_population = genetic_algorithm.create_next_generation(population)

                    print(f"Generation: {current_generation}, Best Fitness: {fittest_individual.fitness}")
                    save_generation_data(fittest_individual, population, current_generation, ga_uuid_string)

                    #ga_history.append(fittest_individual)
                    population = new_population
                    current_generation += 1

                    # send nuevos pesos a main2
                    weights = population[0].get_weights()
                    load_robot_weights(robot_network, weights)

                    robot.step(timestep)
                    if current_generation == GENERATIONS:
                        pass

            robot2_inputs = torch.zeros(9) # 9 inputs: 8 light + 1 seeing object

            if robot1_receiver.getQueueLength() > 0:
                robot2_inputs = torch.tensor(json.loads(robot1_receiver.getString()))
                robot1_receiver.nextPacket()

            # obtener valores de los sensores
            robot1_light_sensor_values = get_sensor_values(light_sensors)
            robot1_normalized_light_sensor_values = normalize_sensor_values(robot1_light_sensor_values, 0, 4095)
            robot1_discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in robot1_normalized_light_sensor_values]
            robot1_image = get_np_image_from_camera(camera)
            robot1_average_color = calculate_average_color(robot1_image)
            robot1_seeing_object = np.logical_and(robot1_average_color[0] > robot1_average_color[1], robot1_average_color[0] > robot1_average_color[2]).astype(int)
            robot1_inputs = torch.cat((torch.tensor(robot1_discretize_light_sensor_values).to(torch.float32), torch.tensor([robot1_seeing_object]).to(torch.float32)))
            
            # print(f"robot1_inputs: {robot1_inputs}")
            # print(f"robot2_inputs: {robot2_inputs}")

            inputs = torch.cat((robot1_inputs, robot2_inputs)).to(torch.float32)

            # obtener direcciones
            outputs = robot_network.forward(inputs)
            robot1_percentage_left_speed = outputs[0].item()
            robot1_percentage_right_speed = outputs[1].item()

            robot2_percentage_left_speed = outputs[2].item()
            robot2_percentage_right_speed = outputs[3].item()
            
            robot1_emitter.send(str([robot2_percentage_left_speed, robot2_percentage_right_speed]))

            left_motor_velocity = robot1_percentage_left_speed * MAX_SPEED
            right_motor_velocity = robot1_percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
    elif robot_name == "robot2":
        robot.step(timestep)

        robot2_node = robot.getFromDef("ROBOT2")

        robot2_emitter = robot.getDevice("emitter")

        robot2_receiver = robot.getDevice("receiver")
        robot2_receiver.enable(timestep)

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
            
            # obtener valores de los sensores
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image).tolist()
            seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)
            robot2_emitter.send(str(discretize_light_sensor_values + [seeing_object]))

            directions = [0, 0]
            if robot2_receiver.getQueueLength() > 0:
                directions = json.loads(robot2_receiver.getString())
                robot2_receiver.nextPacket()

            left_motor_velocity = directions[0] * MAX_SPEED
            right_motor_velocity = directions[1] * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

    weights = last_generation_info["fittest_individual_weights"]
    load_robot_weights(robot_network, weights)
    

elif mode == Mode.REPLAY:

    if robot_name == "robot1":
        history_uuid = "46c4c704-bbaa-4a16-9be1-0e57f74425fd"
        config, gens_info = get_history_info(history_uuid)


        robot1_node = robot.getFromDef("ROBOT1")
        robot2_node = robot.getFromDef("ROBOT2")

        object_node = robot.getFromDef("OBJECT")
        translation_field_object = object_node.getField("translation")
        rotation_field_object = object_node.getField("rotation")

        translation_field_robot1 = robot1_node.getField("translation")
        rotation_field_robot1 = robot1_node.getField("rotation")

        translation_field_robot2 = robot2_node.getField("translation")
        rotation_field_robot2 = robot2_node.getField("rotation")

        translation_field_object = object_node.getField("translation")
        rotation_field_object = object_node.getField("rotation")

        INITIAL_POSITION_ROBOT1 = config["initial_position_robot1"]
        INITIAL_ROTATION_ROBOT1 = config["initial_rotation_robot1"]

        INITIAL_POSITION_ROBOT2 = config["initial_position_robot2"]
        INITIAL_ROTATION_ROBOT2 = config["initial_rotation_robot2"]

        INITIAL_POSITION_OBJECT = config["initial_position_object"]
        INITIAL_ROTATION_OBJECT = config["initial_rotation_object"]

        translation_field_robot1.setSFVec3f(INITIAL_POSITION_ROBOT1)
        rotation_field_robot1.setSFRotation(INITIAL_ROTATION_ROBOT1)

        translation_field_robot2.setSFVec3f(INITIAL_POSITION_ROBOT2)
        rotation_field_robot2.setSFRotation(INITIAL_ROTATION_ROBOT2)

        translation_field_object.setSFVec3f(INITIAL_POSITION_OBJECT)
        rotation_field_object.setSFRotation(INITIAL_ROTATION_OBJECT)

        robot1_receiver = robot.getDevice("receiver")
        robot1_receiver.enable(timestep)

        robot1_emitter = robot.getDevice("emitter")

        robot_network = SimpleRobotNetwork()

        current_generation = 0
        current_time = 0
        previous_time = 0

        weights = gens_info[current_generation]["fittest_individual_weights"]
        load_robot_weights(robot_network, weights)
    
        print(f"replaying {current_generation}/{len(gens_info)} ({gens_info[current_generation]['fittest_individual_fitness']})")
        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time:
                current_generation += 1

                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                robot1_node.resetPhysics()
                robot2_node.resetPhysics()
                object_node.resetPhysics()

                translation_field_robot1.setSFVec3f(INITIAL_POSITION_ROBOT1)
                rotation_field_robot1.setSFRotation(INITIAL_ROTATION_ROBOT1)

                translation_field_robot2.setSFVec3f(INITIAL_POSITION_ROBOT2)
                rotation_field_robot2.setSFRotation(INITIAL_ROTATION_ROBOT2)

                translation_field_object.setSFVec3f(INITIAL_POSITION_OBJECT)
                rotation_field_object.setSFRotation(INITIAL_ROTATION_OBJECT)

                if current_generation < len(gens_info):
                    weights = gens_info[current_generation]["fittest_individual_weights"]
                    load_robot_weights(robot_network, weights)

                    print(f"replaying {current_generation}/{len(gens_info)} ({gens_info[current_generation]['fittest_individual_fitness']})")
                else:
                    break
                
            robot2_inputs = torch.zeros(11) # 11 inputs: 8 light sensors + 3 average color

            if robot1_receiver.getQueueLength() > 0:
                robot2_inputs = torch.tensor(json.loads(robot1_receiver.getString()))
                robot1_receiver.nextPacket()

            # obtener valores de los sensores
            robot1_light_sensor_values = get_sensor_values(light_sensors)
            robot1_normalized_light_sensor_values = normalize_sensor_values(robot1_light_sensor_values, 0, 4095)
            robot1_image = get_np_image_from_camera(camera)
            robot1_average_color = calculate_average_color(robot1_image)
            robot1_inputs = torch.cat((torch.tensor(robot1_normalized_light_sensor_values), torch.tensor(robot1_average_color)))

            inputs = torch.cat((robot1_inputs, robot2_inputs))

            # obtener direcciones
            outputs = robot_network.forward(inputs)
            robot1_percentage_left_speed = outputs[0].item()
            robot1_percentage_right_speed = outputs[1].item()

            robot2_percentage_left_speed = outputs[2].item()
            robot2_percentage_right_speed = outputs[3].item()
            
            robot1_emitter.send(str([robot2_percentage_left_speed, robot2_percentage_right_speed]))

            left_motor_velocity = robot1_percentage_left_speed * MAX_SPEED
            right_motor_velocity = robot1_percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
    elif robot_name == "robot2":

        robot2_node = robot.getFromDef("ROBOT2")

        robot2_receiver = robot.getDevice("receiver")
        robot2_receiver.enable(timestep)

        robot2_emitter = robot.getDevice("emitter")

        current_time = 0
        previous_time = 0

        while robot.step(timestep) != -1:
            previous_time = current_time
            current_time = robot.getTime() % MAX_TIME

            if previous_time > current_time:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image).tolist()

            robot2_emitter.send(str(normalized_light_sensor_values + average_color))

            directions = [0, 0]
            if robot2_receiver.getQueueLength() > 0:
                directions = json.loads(robot2_receiver.getString())
                robot2_receiver.nextPacket()

            left_motor_velocity = directions[0] * MAX_SPEED
            right_motor_velocity = directions[1] * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)

elif mode == Mode.EXECUTION:
    if robot_name == "robot1":
        ga_uuid_string = "3a35ba03-1d7f-4208-874b-acd899e26b32"
        generation_file = "generation_25.json"

        generation_data = read_json_to_dict(f"histories/{ga_uuid_string}/{generation_file}")

        print(f"Generation: {generation_data['generation']}, Best Fitness: {generation_data['fittest_individual_fitness']}")

        weights = generation_data["fittest_individual_weights"]

        robot_network = SimpleRobotNetwork()
        load_robot_weights(robot_network, weights)

        robot1_receiver = robot.getDevice("receiver")
        robot1_receiver.enable(timestep)

        robot1_emitter = robot.getDevice("emitter")

        while robot.step(timestep) != -1:

            robot2_inputs = torch.zeros(9)

            if robot1_receiver.getQueueLength() > 0:
                robot2_inputs = torch.tensor(json.loads(robot1_receiver.getString()))
                robot1_receiver.nextPacket()

            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            robot1_discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]
            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image)
            robot1_seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)

            robot1_inputs = torch.cat((torch.tensor(robot1_discretize_light_sensor_values).to(torch.float32), torch.tensor([robot1_seeing_object]).to(torch.float32)))

            inputs = torch.cat((robot1_inputs, robot2_inputs)).to(torch.float32)

            outputs = robot_network.forward(inputs)
            robot1_percentage_left_speed = outputs[0].item()
            robot1_percentage_right_speed = outputs[1].item()

            robot2_percentage_left_speed = outputs[2].item()
            robot2_percentage_right_speed = outputs[3].item()

            robot1_emitter.send(str([robot2_percentage_left_speed, robot2_percentage_right_speed]))
            
            left_motor_velocity = robot1_percentage_left_speed * MAX_SPEED
            right_motor_velocity = robot1_percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
    elif robot_name == "robot2":
        robot.step(timestep)

        robot2_node = robot.getFromDef("ROBOT2")

        robot2_receiver = robot.getDevice("receiver")
        robot2_receiver.enable(timestep)

        robot2_emitter = robot.getDevice("emitter")

        while robot.step(timestep) != -1:
            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
            discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image)
            robot2_seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)

            robot2_emitter.send(str(discretize_light_sensor_values + [robot2_seeing_object]))

            directions = [0, 0]
            if robot2_receiver.getQueueLength() > 0:
                directions = json.loads(robot2_receiver.getString())
                robot2_receiver.nextPacket()
            
            left_motor_velocity = directions[0] * MAX_SPEED
            right_motor_velocity = directions[1] * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
elif mode == Mode.TESTING:
    print("TESTING MODE")

    if robot_name == "robot1":
        robot_node = robot.getFromDef("ROBOT1")

        translation_field_robot = robot_node.getField("translation")
        rotation_field_robot = robot_node.getField("rotation")

        object_node = robot.getFromDef("OBJECT")
        translation_field_object = object_node.getField("translation")
        rotation_field_object = object_node.getField("rotation")

        area_node = robot.getFromDef("AREA")
        translation_field_area = area_node.getField("translation")

        INITIAL_POSITION_ROBOT = translation_field_robot.getSFVec3f()
        INITIAL_ROTATION_ROBOT = rotation_field_robot.getSFRotation()

        INITIAL_POSITION_OBJECT = translation_field_object.getSFVec3f()
        INITIAL_ROTATION_OBJECT = rotation_field_object.getSFRotation()

        INITIAL_POSITION_AREA = translation_field_area.getSFVec3f()

        robot_network = RobotNetwork()

        max_test_time = 10
        current_test_time = 0
        previous_test_time = 0

        for name, param in robot_network.named_parameters():
            print(f"{name}: mean = {param.mean().item()}, std = {param.std().item()}")

        test = 0
        fitness_ = 0
        while robot.step(timestep) != -1:
            previous_test_time = current_test_time
            current_test_time = robot.getTime() % max_test_time

            if previous_test_time > current_test_time:
                fitness_ = fitness(translation_field_object.getSFVec3f(), INITIAL_POSITION_AREA)
                test += 1

                translation_field_robot.setSFVec3f(INITIAL_POSITION_ROBOT)
                rotation_field_robot.setSFRotation(INITIAL_ROTATION_ROBOT)

                robot_node.resetPhysics()

                robot.step(timestep)

                print(f"Reseteando simulación [{robot_name}] ({test}) - Fitness: {fitness_}")

                fitness_ = 0

            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(average_color)))

            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            
            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)
            
    elif robot_name == "robot2":
        robot_node = robot.getFromDef("ROBOT2")

        translation_field_robot = robot_node.getField("translation")
        rotation_field_robot = robot_node.getField("rotation")

        INITIAL_POSITION_ROBOT = translation_field_robot.getSFVec3f()
        INITIAL_ROTATION_ROBOT = rotation_field_robot.getSFRotation()

        robot_network = RobotNetwork()

        max_test_time = 10
        current_test_time = 0
        previous_test_time = 0

        test = 0
        while robot.step(timestep) != -1:
            previous_test_time = current_test_time
            current_test_time = robot.getTime() % max_test_time

            if previous_test_time > current_test_time:
                test += 1

                translation_field_robot.setSFVec3f(INITIAL_POSITION_ROBOT)
                rotation_field_robot.setSFRotation(INITIAL_ROTATION_ROBOT)

                robot_node.resetPhysics()

                robot.step(timestep)

                print(f"Reseteando simulación [{robot_name}] ({test})")

            light_sensor_values = get_sensor_values(light_sensors)
            normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)

            image = get_np_image_from_camera(camera)
            average_color = calculate_average_color(image)

            normalized_sensor_values = torch.cat((torch.tensor(normalized_light_sensor_values), torch.tensor(average_color)))

            outputs = robot_network.forward(normalized_sensor_values)
            percentage_left_speed = outputs[0].item()
            percentage_right_speed = outputs[1].item()
            
            left_motor_velocity = percentage_left_speed * MAX_SPEED
            right_motor_velocity = percentage_right_speed * MAX_SPEED

            left_motor.setVelocity(left_motor_velocity)
            right_motor.setVelocity(right_motor_velocity)