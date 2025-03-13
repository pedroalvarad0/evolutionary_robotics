"""manual_controller controller."""

from controller import Robot, Keyboard, Supervisor
import numpy as np
import json

def get_sensor_values(sensors):
    sensor_values = []
    for sensor in sensors:
        sensor_values.append(sensor.getValue())
    return sensor_values

def get_np_image_from_camera(camera):
    image = camera.getImage()
    # Obtener dimensiones de la imagen
    width = camera.getWidth()
    height = camera.getHeight()

    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    image_rgb = image_array[:, :, :3][:, :, ::-1]  # Convertir BGRA a RGB
    # print("Forma de la imagen:", image_array.shape)

    return image_rgb

def normalize_rotation_angle(rotation_angle):
    if rotation_angle < 0:
        rotation_angle += 2 * np.pi
    return rotation_angle

def calculate_average_color(image):
    """
    Calcula el color promedio de una imagen RGB.
    
    Args:
        image: Array numpy de forma (height, width, 3) con valores RGB
        
    Returns:
        tuple: Color promedio en formato (R, G, B)
    """
    # Promedia los valores a lo largo de los ejes height y width
    avg_color = np.mean(image, axis=(0,1))
    # Redondea los valores a enteros
    avg_color = np.round(avg_color).astype(int)
    return tuple(avg_color)

# Create the Robot instance
robot = Supervisor()
robot_name = robot.getName()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Initialize the keyboard
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

robot_node = robot.getFromDef("MAIN2")
rotation_field = robot_node.getField('rotation')

# Constants for motor speeds
MAX_SPEED = 6.28  # Maximum speed for the motors
TURN_COEFFICIENT = 0.5  # Coefficient to reduce speed while turning

distance_sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
distance_sensors = []

for i in range(len(distance_sensor_names)):
    sensor = robot.getDevice(distance_sensor_names[i])
    sensor.enable(timestep)
    distance_sensors.append(sensor)


communication = robot.getDevice("receiver")
communication.enable(timestep)

# Main loop
while robot.step(timestep) != -1:
    # Get the pressed key
    key = keyboard.getKey()

    #sensor_values = get_sensor_values(distance_sensors)
    
    # Initialize motor speeds
    left_speed = 0.0
    right_speed = 0.0
    
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

    rotation_angle = rotation_field.getSFRotation()[3]
    normalized_rotation_angle = normalize_rotation_angle(rotation_angle)

    if communication.getQueueLength() > 0:
        light_finder_position = json.loads(communication.getString())
        print(f"{light_finder_position}")
        communication.nextPacket()

    # image = camera.getImage()
    # image_rgb = get_np_image_from_camera(camera)
    # avg_camera_color = calculate_average_color(image_rgb)
    # print(f"avg_camera_color({robot_name}):", avg_camera_color)

