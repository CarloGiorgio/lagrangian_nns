import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.nn import functional as F


import torchsummary

import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


import os
import glob



class Network(nn.Module):
    def __init__(self,input,hidden,output):
        super(Network,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input,hidden),
            nn.Softplus()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.Softplus()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden,output)
        )      
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out