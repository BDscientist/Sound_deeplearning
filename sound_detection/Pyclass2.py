import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
import data_provider2




class my_datset2(Dataset):
    
    
    def __init__(self,train = True):
        
        
        self.train = train

        total_train = np.load('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/Final_train_dataset.npz')
        total_test = np.load('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/Final_test_dataset.npz')
        
                      
   
        if self.train :
            
            x=np.array([0,0,0,0,0,0,0])
            self.train_data = np.zeros((6723,200))
            train_data ,self.train_label = total_train['X'], total_train['y']
              
            
            for i in range(len(self.train_data)):
                self.train_data[i,] =  np.concatenate((train_data[i,],x),axis=None)
            
            
            self.train_data =self.train_data.reshape(6723,1,10,20)
            dtype = torch.FloatTensor
            self.train_data  = torch.as_tensor(self.train_data).type(dtype)
            
            
            
           
            self.train_label = np.array(self.train_label)
            
            self.train_label = torch.tensor(self.train_label, dtype=torch.long)


                
        else :
            
            
            x=np.array([0,0,0,0,0,0,0])
            self.test_data = np.zeros((1529,200))
            test_data ,self.test_label = total_test['X'], total_test['y']
              
            
            for i in range(len(self.test_data)):
                self.test_data[i,] =  np.concatenate((test_data[i,],x),axis=None)
            
            
            self.test_data =self.test_data.reshape(1529,1,10,20)
            dtype = torch.FloatTensor
            self.test_data  = torch.as_tensor(self.test_data).type(dtype)
            
            
            
           
            self.test_label = np.array(self.test_label)
            
            self.test_label = torch.tensor(self.test_label, dtype=torch.long)
 
    
    
    
    def __getitem__(self,idx):
            
        if self.train :
            
            data , target = self.train_data[idx], self.train_label[idx]
            
        else:
                
            data, target = self.test_data[idx], self.test_label[idx]
                
        
        
        return data, target
    

                
                
                
                
    
    def __len__(self):
        
            
        if self.train:
            return len(self.train_data)
            
        
        else:
            return len(self.test_data)
            

    
    
    