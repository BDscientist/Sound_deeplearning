import numpy as np
import random
    
    
    
def prepare_data():
    
        
    global train_samples, test_samples

    train_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_train_samples.txt').read().strip().split('\n')
         

    test_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_test_samples.txt').read().strip().split('\n')
 

  
def get_random_sample(part):
    
    global train_samples, test_samples
    
    option = 'spectrum_Stft'
    
    train_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_train_samples.txt').read().strip().split('\n')
   

    test_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_test_samples.txt').read().strip().split('\n')
   
      
    if part == 'new_train':
        
        samples = train_samples
            
        
    elif part == 'new_test':
        
        samples = test_samples
            
    else :
        
        print('Please use train, valid, or test for the part name')

    i = random.randrange(len(samples))
    spectrum = np.load('/home/libedev/mute/mute-hero/download/dataset/'
                           +str(part)+'/'+str(option)+'/'+samples[i]+'.npy')
        
        
    return spectrum        

    
    
def get_random_batch(part):
    
    
    X = np.zeros((400, 188, 513, 1))
    
    option = 'spectrum_Stft'
    
    for b in range(0,399):
        
        s = get_random_sample(part)
        
        X[b, :, :, 0] = s[:188, :513]
        
    
    return X


