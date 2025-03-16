import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotNetwork(nn.Module):
    def __init__(self, input_size=16, hidden_size=10, output_size=3):
        super(RobotNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x