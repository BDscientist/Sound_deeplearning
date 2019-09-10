import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from random import *
import time

class data_cut:
    
    def __init__(self,base,base2,second):
        self.base = base
        # base :  음원이 적재되어있는 주소
        #'/home/libedev/mute/mute-hero/download/freesound/airplane/'
    
        self.base2 = base2
        # base2 : 음원을 자르고 다시 재 적재할 주소
    
        self.second = int(second)
        # second :  몇 초로자를것인지
    

    def data_process(self):   
        
        a=[]

        for filename in os.listdir(str(self.base)):
     
            a.append(filename)

    
        for i in a:
            print(i)
            
            count = 0
            start = 0   
            
            y, sr = librosa.load(self.base+i)
            end = sr*self.second
            criteria = int(y.shape[0]) / int(sr)    
            criteria = int(criteria) 
            print(criteria)
            
        
            for j in range(0,criteria-self.second):
                     
                new_y2 = y[start :round(end)]
                
                librosa.output.write_wav(str(self.base2)+'_'+str(count)+'.wav', new_y2, sr) 
                # save half-cut file
                #'/home/libedev/mute/mute-hero/download/cut_file'
                
                print("finish")
                
                start += sr
                end += sr
                count +=1
    
a= input("적재되어져 있는 파일의 주소를 입력하시오: ")
# ex    /home/libedev/mute/mute-hero/download/train/Conversation/ 

b= input("작업 된 파일들의 적재장소를 입력하시오: ")
#ex     /home/libedev/mute/mute-hero/download/cut_file 

c= input("몇 s(초)로 구간을 자를 것인지 입력하시오: ")

s= data_cut(a,b,c)
s.data_process()