"""area_finder_controller.py"""

from controller import Supervisor, Keyboard  
import numpy as np
robot = Supervisor()
robot_name = robot.getName()

timestep = int(robot.getBasicTimeStep())

keyboard = Keyboard()
keyboard.enable(timestep)

light = robot.getFromDef("LIGHT")

# Initialize the motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Set the target position to infinity (speed control)
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set the initial velocity to 0
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Initialize the camera
camera = robot.getDevice('camera1')
camera.enable(timestep)

led0 = robot.getDevice("led0")

#print(light.getField("on").getSFBool())

while robot.step(timestep) != -1:
    
    if np.random.rand() < 0.80:
        light.getField("on").setSFBool(True)
    else:
        light.getField("on").setSFBool(False)

    
