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
import Pyclass
import Pyclass2



from torch.utils.data import Dataset, DataLoader


sound_train = Pyclass2.my_datset2(train = True)
                         

sound_test = Pyclass2.my_datset2(train = False)



batch_size = 50

train_loader  = DataLoader(dataset=sound_train,
                                           batch_size=batch_size,
                                           shuffle=False
                                           ,num_workers=1)

test_loader = DataLoader(dataset=sound_test,
                                         batch_size=batch_size,
                                         shuffle=False,
                        num_workers=1)
                        
                        
                        
                        
print(sound_train.train_data.shape)
print(sound_train.train_label.shape)
print(sound_train.train_data.dtype)

class CNN(nn.Module):
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=16,kernel_size = 3),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            
        )
        
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*1,1000),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,11),
            nn.Softmax()
        )       
        
    def forward(self,x):
        print(x.data.shape)
        out = self.layer(x)
        out = out.view(out.shape[0],-1)
        out = self.fc_layer(out)

        return out
        
        
model = CNN().cuda()

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5000

save_path = '/home/libedev/mute/mute-hero/download/dataset/model2/'
model_path = save_path + 'model.pkl'
if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model Loaded!")

else:
    
    for epoch in range(num_epochs):

        total_batch = len(sound_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.cuda()
            Y = batch_labels.cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    if not os.path.isfile(model_path):
        print("Model Saved!")
        torch.save(model.state_dict(), model_path)



model.eval()

correct = 0
total = 0

for sound, labels in test_loader:
    
    sound = sound.cuda()
    outputs = model(sound)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Accuracy of test sound: %f %%' % (100 * float(correct) / total))

