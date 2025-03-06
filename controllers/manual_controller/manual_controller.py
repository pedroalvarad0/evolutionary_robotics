"""manual_controller controller."""

from controller import Robot, Keyboard

# Create the Robot instance
robot = Robot()

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

# Constants for motor speeds
MAX_SPEED = 6.28  # Maximum speed for the motors
TURN_COEFFICIENT = 0.5  # Coefficient to reduce speed while turning

# Main loop
while robot.step(timestep) != -1:
    # Get the pressed key
    key = keyboard.getKey()
    
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

