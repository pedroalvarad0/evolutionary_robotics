"""save_light_path controller."""

from controller import Robot, Supervisor
import numpy as np

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# nodes
light_node = robot.getFromDef("LIGHT")

light_translation_field = light_node.getField("translation")

MAX_TIME = 30
time_before_capture = 5
light_positions = []

while robot.step(timestep) != -1:
    
    # Determinar la etapa actual basada en el tiempo
    if robot.getTime() < time_before_capture:
        print("preparando...")
            
    elif robot.getTime() < MAX_TIME + time_before_capture:
        print("capturando...")
        light_positions.append(light_translation_field.getSFVec3f())
    else:
        print("Captura finalizada")
        light_positions_array = np.array(light_positions)
        np.save("light_positions.npy", light_positions_array)
        break


