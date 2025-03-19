"""general_move_box_controller controller."""
from controller import Supervisor

MAX_SPEED = 6.28

robot = Supervisor()
robot_name = robot.getName()
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

epuck1_node = robot.getFromDef("EPUCK1")
epuck2_node = robot.getFromDef("EPUCK2")
object_node = robot.getFromDef("OBJECT")
arena_node = robot.getFromDef("ARENA")

fields = {
    "epuck1": {
        "translation": epuck1_node.getField("translation"),
        "rotation": epuck1_node.getField("rotation")
    },
    "epuck2": {
        "translation": epuck2_node.getField("translation"),
        "rotation": epuck2_node.getField("rotation")
    },
    "object": {
        "translation": object_node.getField("translation"),
        "rotation": object_node.getField("rotation")
    },
    "arena": {
        "translation": arena_node.getField("translation"),
        "floorSize": arena_node.getField("floorSize")
    }
}

