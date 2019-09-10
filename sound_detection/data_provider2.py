import numpy as np
import random



def prepare_data():
    
        
    global train_samples, train_labels, test_samples, test_labels

    train_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_train_samples.txt').read().strip().split('\n')
    #train_labels = [int(label) for label in open('/home/libedev/mute/mute-hero/download/dataset/new_train_labels.txt').read().strip().split('\n')]   
    
    train_labels = np.genfromtxt('/home/libedev/mute/mute-hero/download/dataset/new_train_labels.txt', encoding='ascii',dtype=int)

    test_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_test_samples.txt').read().strip().split('\n')
    #test_labels = [int(label) for label in open('/home/libedev/mute/mute-hero/download/dataset/new_test_labels.txt').read().strip().split('\n')]
    
    test_labels=np.genfromtxt('/home/libedev/mute/mute-hero/download/dataset/new_test_labels.txt', encoding='ascii',dtype=int)


    
def get_random_sample(part,option):
    
    
    global train_samples, train_labels, test_samples, test_labels
    
    
    
    train_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_train_samples.txt').read().strip().split('\n')
    #train_labels = [int(label) for label in open('/home/libedev/mute/mute-hero/download/dataset/new_train_labels.txt').read().strip().split('\n')]     

    train_labels = np.genfromtxt('/home/libedev/mute/mute-hero/download/dataset/new_train_labels.txt', encoding='ascii',dtype=int)

    test_samples = open('/home/libedev/mute/mute-hero/download/dataset/new_test_samples.txt').read().strip().split('\n')
    #test_labels = [int(label) for label in open('/home/libedev/mute/mute-hero/download/dataset/new_test_labels.txt').read().strip().split('\n')]

    test_labels = np.genfromtxt('/home/libedev/mute/mute-hero/download/dataset/new_test_labels.txt', encoding='ascii',dtype=int)
      
    if part == 'new_train':
        
        samples = train_samples
        labels = train_labels   
        
    elif part == 'new_test':
        
        samples = test_samples
        labels = test_labels
        
    else :
        
        print('Please use train, valid, or test for the part name')

    i = random.randrange(len(samples))
    spectrum = np.load('/home/libedev/mute/mute-hero/download/dataset/'
                           +str(part)+'/'+str(option)+'/'+samples[i]+'.npy')
        
        
    return spectrum, labels[i]        

    
    
def get_random_batch(part,option):
    
    
    global train_samples, train_labels, test_samples, test_labels
    
    #option = 'spectrum_Stft'

    if part == 'new_train':
        
        data_amount = len(train_samples)
        example_data = np.load('/home/libedev/mute/mute-hero/download/dataset/'
                               +str(part)+'/'+str(option)+'/'+train_samples[0]+'.npy')
    else :
        
        data_amount = len(test_samples)
        example_data = np.load('/home/libedev/mute/mute-hero/download/dataset/'
                               +str(part)+'/'+str(option)+'/'+test_samples[0]+'.npy')
    
    
    X = np.zeros((data_amount, example_data.shape[0], example_data.shape[1], 1))
    Y = np.zeros((data_amount,))
    
    for i in range(0,data_amount):
        
        s,l = get_random_sample(part,option)
        
        X[i, :, :, 0] = s[:example_data.shape[0], :example_data.shape[1]]
        Y[i] =  l
        #Y = Y.astype('int16')
        f_labels = np.save('/home/libedev/mute/mute-hero/download/dataset/'+str(option)+'.npy',Y)
        new_Y = np.load('/home/libedev/mute/mute-hero/download/dataset/'+str(option)+'.npy')
        
    return X,new_Y


