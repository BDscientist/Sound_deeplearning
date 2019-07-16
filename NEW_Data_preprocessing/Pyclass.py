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
    
    
    def __init__(self,part, option, train = True):
        
        self.part = part
        self.option = option
        self.train = train

        data_provider2.prepare_data()
        
                      
   
        if self.train :
            
            
            
            self.train_data ,self.train_label = data_provider2.get_random_batch(part,option) 
              
          
                      
            self.train_data = self.train_data.transpose((0,3,1,2))
            
            dtype = torch.FloatTensor
            self.train_data = torch.as_tensor(self.train_data).type(dtype)
            self.train_data = torch.as_tensor(self.train_data)
            
            
            #self.train_label = torch.as_tensor(np.array(self.train_label))
            self.train_label = np.array(self.train_label)
            
            self.train_label = torch.tensor(self.train_label, dtype=torch.long)


                
        else :
            
            
            self.test_data ,self.test_label = data_provider2.get_random_batch(part,option)
            
            
            
            
            
            self.test_data = self.test_data.transpose((0,3,1,2))
            
            dtype = torch.FloatTensor
            self.test_data = torch.as_tensor(self.test_data).type(dtype) 
            self.test_data = torch.as_tensor(self.test_data)
            
            
            #self.test_label = torch.as_tensor(np.array(self.test_label))
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
            

    
    
    