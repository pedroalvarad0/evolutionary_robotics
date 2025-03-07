"""multi_robot_controller controller."""

from controller import Robot, Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
