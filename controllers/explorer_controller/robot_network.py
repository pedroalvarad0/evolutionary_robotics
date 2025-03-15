import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=5, output_size=3):
        super(RobotNetwork, self).__init__()
        # Capa de entrada (8 sensores) a capa oculta
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Capa oculta a capa de salida (2 motores)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Aplicamos ReLU como función de activación en la capa oculta
        x = F.relu(self.fc1(x))
        # La capa de salida usa tanh para obtener valores entre -1 y 1
        # Esto nos da velocidades positivas y negativas para los motores
        x = torch.sigmoid(self.fc2(x))
        return x