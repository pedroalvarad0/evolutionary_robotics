import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRobotNetwork(nn.Module):
    def __init__(self, input_size=9, output_size=2):
        super(SimpleRobotNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x