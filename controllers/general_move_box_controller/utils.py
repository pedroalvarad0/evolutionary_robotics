import torch
import numpy as np
import os
import json

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
    return avg_color.astype(np.float32)


def load_robot_weights(robot_network, weights):
    idx = 0
    for param in robot_network.parameters():
        layer_size = param.data.numel()
        layer_weights = weights[idx:idx + layer_size]
        param.data = torch.tensor(layer_weights).reshape(param.data.shape)
        idx += layer_size


class Position:
    def __init__(self, x, y, rotation=0):
        self.x = x
        self.y = y
        self.rotation = rotation
    
    def distance_to(self, other):
        """Calcula la distancia euclidiana a otra posición"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self):
        """Convierte la posición a una tupla (x, y, rotation)"""
        return (self.x, self.y, self.rotation)

class AreaSampler:
    def __init__(self, center, size, margin=0.2, min_distance=0.3):
        self.center = center
        self.size = size
        self.margin = margin
        self.min_distance = min_distance
        
        # Calcula los límites efectivos del área
        self.half_width = size[0] / 2
        self.half_height = size[1] / 2
        self.x_min = center[0] - self.half_width + (self.half_width * margin)
        self.x_max = center[0] + self.half_width - (self.half_width * margin)
        self.y_min = center[1] - self.half_height + (self.half_height * margin)
        self.y_max = center[1] + self.half_height - (self.half_height * margin)
        
        # Parámetros para la distribución normal del objeto
        self.sigma_x = (self.half_width * (1 - margin)) / 3
        self.sigma_y = (self.half_height * (1 - margin)) / 3
    
    def sample_uniform(self):
        """Genera una posición uniforme (para robots)"""
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        rotation = np.random.uniform(-np.pi, np.pi)
        return Position(x, y, rotation)
    
    def sample_normal(self):
        """Genera una posición normal (para objetos)"""
        x = np.clip(
            np.random.normal(self.center[0], self.sigma_x),
            self.x_min, self.x_max
        )
        y = np.clip(
            np.random.normal(self.center[1], self.sigma_y),
            self.y_min, self.y_max
        )
        rotation = np.random.uniform(-np.pi, np.pi)
        return Position(x, y, rotation)
    
    def generate_valid_configuration(self):
        """Genera una configuración válida de posiciones"""
        max_attempts = 50
        positions = []
        
        # Genera posiciones para robots (uniforme)
        for _ in range(2):
            for _ in range(max_attempts):
                pos = self.sample_uniform()
                if all(pos.distance_to(p) >= self.min_distance for p in positions):
                    positions.append(pos)
                    break
            else:
                return None  # No se encontró configuración válida
        
        # Genera posición para objeto (normal)
        for _ in range(max_attempts):
            pos = self.sample_normal()
            if all(pos.distance_to(p) >= self.min_distance for p in positions):
                positions.append(pos)
                break
        else:
            return None  # No se encontró configuración válida
        
        return {
            "epuck1": positions[0].to_tuple(),
            "epuck2": positions[1].to_tuple(),
            "object": positions[2].to_tuple()
        }
    
def create_config_file(
    ga_uuid,
    max_time,
    population_size,
    generations,
    crossover_rate,
    mutation_rate,
    representation,
    tests_per_individual):
    os.makedirs(f"histories/{str(ga_uuid)}", exist_ok=True)
    
    config_file = {
        "ga_uuid": str(ga_uuid),
        "max_time": max_time,
        "population_size": population_size,
        "generations": generations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "representation": representation,
        "tests_per_individual": tests_per_individual
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
    

def get_history_info(history_uuid):
    history_path = os.path.join("histories", history_uuid)

    config = read_json_to_dict(os.path.join(history_path, "config.json"))

    files = sorted(os.listdir(history_path), 
               key=lambda x: os.path.getctime(os.path.join(history_path, x)))
    
    gens_info = []
    for i in range(1, len(files)):
        file_path = os.path.join(history_path, files[i])
        gen_dict = read_json_to_dict(file_path)
        #del gen_dict["population"]
        gens_info.append(gen_dict)

    return config, gens_info


def get_last_generation_info(ga_uuid):
    history_path = os.path.join("histories", ga_uuid)
    files = sorted(os.listdir(history_path), 
               key=lambda x: os.path.getctime(os.path.join(history_path, x)))
    
    last_generation_file = os.path.join(history_path, files[-1])
    return read_json_to_dict(last_generation_file)