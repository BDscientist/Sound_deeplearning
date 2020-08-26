import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
import os
import PYCLASS2_air_Copy1
from torch.utils.data import Dataset, DataLoader


class CNN2(nn.Module):
    def __init__(self):
        
        super(CNN2, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=16,kernel_size = 3),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2,2)
            
        )
        
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*6*1,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,3),
            nn.ReLU(),
            nn.Softmax(dim=1)
            #nn.Softmax()
        )       
        
    def forward(self,x):
        print(x.data.shape)
        out = self.layer(x)
        out = out.view(out.shape[0],-1)
        out = self.fc_layer(out)

        return out

