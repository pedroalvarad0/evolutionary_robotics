from controller import Supervisor, Keyboard
import numpy as np

robot = Supervisor()
robot_name = robot.getName()

timestep = int(robot.getBasicTimeStep())

keyboard = Keyboard()
keyboard.enable(timestep)

# Initialize the motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Set the target position to infinity (speed control)
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set the initial velocity to 0
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

light_sensor_names = ["ls0", "ls1", "ls2", "ls3", "ls4", "ls5", "ls6", "ls7"]
light_sensors = []

for i in range(len(light_sensor_names)):
    sensor = robot.getDevice(light_sensor_names[i])
    sensor.enable(timestep)
    light_sensors.append(sensor)

def get_sensor_values(sensors):
    return [sensor.getValue() for sensor in sensors]

MAX_SPEED = 6.28  # Maximum speed for the motors
TURN_COEFFICIENT = 0.5  # Coefficient to reduce speed while turning

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


camera = robot.getDevice("camera")
camera.enable(timestep)

while robot.step(timestep) != -1:
    sensor_values = get_sensor_values(light_sensors)
    #print(f"sensor_values: {sensor_values}")

    key = keyboard.getKey()

    left_speed = 0.0
    right_speed = 0.0

    image = get_np_image_from_camera(camera)
    average_color = calculate_average_color(image)
    print(f"average_color: {average_color}")
    
    # Handle keyboard input
    if key == Keyboard.UP:
        # Move forward
        left_speed = MAX_SPEED
        right_speed = MAX_SPEED
    elif key == Keyboard.DOWN:
        # Move backward
        left_speed = -MAX_SPEED
        right_speed = -MAX_SPEED
    elif key == Keyboard.LEFT:
        # Turn left
        left_speed = -MAX_SPEED * TURN_COEFFICIENT
        right_speed = MAX_SPEED * TURN_COEFFICIENT
    elif key == Keyboard.RIGHT:
        # Turn right
        left_speed = MAX_SPEED * TURN_COEFFICIENT
        right_speed = -MAX_SPEED * TURN_COEFFICIENT
    
    # Set the motor velocities
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
