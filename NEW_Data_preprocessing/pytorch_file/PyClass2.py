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




class my_datset2(Dataset):
    
    
    
    
    
    def __init__(self, root, option,option2, train = True):
        
        # root -->  new_train , new_test
        self.root = '/home/libedev/mute/mute-hero/download/dataset/'+root+'/'
        
        # option --> 전처리 폴더명
        self.option = option+'/'
        self.option2 = option2
        self.train = train
        
        
        #add = self.root + self.option
        
        all_files = os.path.join(self.root,self.option)
        
        self.data=[os.path.join(dirpath,f)for dirpath, dirnames, filenames in os.walk(all_files) for f in filenames] 
        
        
        
        
        if self.train :
            
            
            
            self.train_data =[]
            
            
            self.train_label=np.load('/home/libedev/mute/mute-hero/download/dataset/'+str(option2)+'.npy')
            
            
            
            for f in self.data:
                
                self.train_data.append(np.load(f))
                
            
          
            self.train_data = np.array(self.train_data)            
            self.train_data = np.array([self.train_data]) 
            dtype = torch.FloatTensor
            self.train_data = self.train_data.transpose((1,0,2,3))
            self.train_data = torch.as_tensor(self.train_data).type(dtype) 
            #self.train_data = self.train_data.squeeze()
            
            self.train_label = torch.as_tensor(np.array(self.train_label))
            
            


                
        else :
            
            
            self.test_data =[]
            #self.test_label =[]
            
            self.test_label = np.load('/home/libedev/mute/mute-hero/download/dataset/new_test_label.npy')
            
            
            
            for f in self.data:
                
                self.test_data.append(np.load(f))

            self.test_data = np.array(self.test_data)                        
            self.test_data = np.array([self.test_data])
            dtype = torch.FloatTensor
            self.test_data = self.test_data.transpose((1,0,2,3))
            self.test_data = torch.as_tensor(self.test_data).type(dtype) 
            #self.test_data = self.test_data.squeeze()
            
            self.test_label = torch.as_tensor(np.array(self.test_label))
 
    
    
    
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
            

    
    
    