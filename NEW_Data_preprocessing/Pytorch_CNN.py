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
import PyClass2
from torch.utils.data import Dataset, DataLoader



sound_train = PyClass2.my_datset2(root='new_train',option='Mel_S_pca',option2 = 'new_train_label',train = True)
                         

sound_test = PyClass2.my_datset2(root='new_test',option='Mel_S_pca',option2 = 'new_test_label',train = False)


batch_size = 10

train_loader  = DataLoader(dataset=sound_train,
                                           batch_size=batch_size,
                                           shuffle=False
                                           ,num_workers=1)

test_loader = DataLoader(dataset=sound_test,
                                         batch_size=batch_size,
                                         shuffle=False,
                        num_workers=1)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=16,kernel_size = 5),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size = 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*6*6,100),
            nn.ReLU(),
            nn.Linear(100,8)
        )       
        
    def forward(self,x):
        print(x.data.shape)
        out = self.layer(x)
        out = out.view(out.shape[0],-1)
        out = self.fc_layer(out)

        return out

    
    
    
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,5),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,5),
            nn.MaxPool2d(2)
    )
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(128,256,7,1,2),
            nn.Conv2d(256,64,5),
            nn.MaxPool2d(2)
    )
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(128,256,5,1,1),
            nn.Conv2d(256,64,5),
            nn.MaxPool2d(2)
    )
        self.layer2_3 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.Conv2d(256,64,5),
            nn.MaxPool2d(2)
    )
        self.fc_layer = nn.Sequential(
            nn.Linear(1*64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,8)
    )       
    
    def forward(self,x):
        print(x.data.shape)
        x= self.layer1(x)
        x1 = self.layer2_1(x)
        x2 = self.layer2_2(x)
        x3 = self.layer2_3(x)
        x=torch.cat((x1,x2,x3),1)
        x= x.view(x.shape[0],-1)
        x=self.fc_layer(x)
        return x   
    
    
    
    
    
model = CNN().cuda()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)



num_epochs = 5
save_path = '/home/libedev/mute/mute-hero/download/dataset/model/'
model_path = save_path + 'model.pkl'


if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    ## model train
    
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

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    if not os.path.isfile(model_path):
        print("Model Saved!")
        torch.save(model.state_dict(), model_path)
        
## model evaluiation
        
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    images = images.cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Accuracy of test sound: %f %%' % (100 * float(correct) / total))