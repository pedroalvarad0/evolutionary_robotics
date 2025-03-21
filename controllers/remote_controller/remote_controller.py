"""remote_controller controller."""

from controller import Robot
import numpy as np
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


robot = Robot()
timestep = int(robot.getBasicTimeStep())

# get robot devices
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

emitter = robot.getDevice("emitter")

receiver = robot.getDevice("receiver")
receiver.enable(timestep)

MAX_SPEED = 6.28

while robot.step(timestep) != -1:
    light_sensor_values = get_sensor_values(light_sensors)
    normalized_light_sensor_values = normalize_sensor_values(light_sensor_values, 0, 4095)
    discretize_light_sensor_values = [1 if x < 0.5 else 0 for x in normalized_light_sensor_values]

    image = get_np_image_from_camera(camera)
    average_color = calculate_average_color(image).tolist()
    seeing_object = np.logical_and(average_color[0] > average_color[1], average_color[0] > average_color[2]).astype(int)

    emitter.send(str(discretize_light_sensor_values + [seeing_object]))

    directions = [0, 0]
    if receiver.getQueueLength() > 0:
        directions = json.loads(receiver.getString())
        receiver.nextPacket()

    left_motor.setVelocity(directions[0] * MAX_SPEED)
    right_motor.setVelocity(directions[1] * MAX_SPEED)