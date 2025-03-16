import numpy as np
import torch
import json
import os

def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values


def normalize_sensor_values(sensor_values, min_value, max_value):
    normalized = [(x - min_value) / (max_value - min_value) for x in sensor_values]
    return normalized


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


def load_robot_weights(robot_network, weights):
    idx = 0
    for param in robot_network.parameters():
        layer_size = param.data.numel()
        layer_weights = weights[idx:idx + layer_size]
        param.data = torch.tensor(layer_weights).reshape(param.data.shape)
        idx += layer_size


def create_config_file(ga_uuid, max_time, population_size, generations):
    os.makedirs(f"histories/{str(ga_uuid)}", exist_ok=True)
    
    config_file = {
        "ga_uuid": str(ga_uuid),
        "max_time": max_time,
        "population_size": population_size,
        "generations": generations
    }

    with open(f"histories/{str(ga_uuid)}/config.json", "w") as f:
        json.dump(config_file, f)


def save_generation_data(fittest_individual, population, current_generation, ga_uuid):
    os.makedirs(f"histories/{str(ga_uuid)}", exist_ok=True)
    
    generation_data = {
        "generation": current_generation,
        "fittest_individual_fitness": fittest_individual.fitness,
        "fittest_individual_weights": fittest_individual.weights,
        "population": []
    }

    for individual_id, individual in enumerate(population):
        generation_data["population"].append({
            "individual_id": individual_id,
            "fitness": individual.fitness,
            "weights": individual.weights
        })

    with open(f"histories/{str(ga_uuid)}/generation_{current_generation}.json", "w") as f:
        json.dump(generation_data, f)


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
