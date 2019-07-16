import numpy as np
import librosa
import os
import sys
import sklearn
from sklearn.decomposition import PCA




class convert:
    
    def extract_Mel_S(part):
        sample_files = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt').read().strip().split('\n')
        frame_length = 0.025 
        frame_stride = 0.010



        if not os.path.exists('/home/libedev/mute/mute-hero/download/dataset/'+part):
            os.mkdir('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum')
        
        for f in sample_files:
        
            print('%d%d: %s'%(sample_files.index(f),len(sample_files),f))
        
            y, sr = librosa.load('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+f+'.wav' ,sr=16000)
        
            input_nfft = int(round(sr*frame_length)) 
            input_stride = int(round(sr*frame_stride))
        
            S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
        
            np.save('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum_Stft/'+f+'.npy', S)
            
            
        
    def extract_Mel_S_PCA(part):
        sample_files = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt').read().strip().split('\n')
        frame_length = 0.025 
        frame_stride = 0.010



        if not os.path.exists('/home/libedev/mute/mute-hero/download/dataset/'+part):
            os.mkdir('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum')
        
        for f in sample_files:
        
            print('%d%d: %s'%(sample_files.index(f),len(sample_files),f))
        
            y, sr = librosa.load('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+f+'.wav' ,sr=16000)
        
            input_nfft = int(round(sr*frame_length)) 
            input_stride = int(round(sr*frame_stride))
        
            S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
            pca = PCA(n_components=10, whiten=True , random_state=0)
            pca.fit(S)
            x_train_pca = pca.transform(S)
        
            np.save('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum_Mel_PCA/'+f+'.npy', x_train_pca)
            
    
    def extract_log_mfcc(part):
    
        sample_files = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt').read().strip().split('\n')

        for f in sample_files:
        
            #if f.split('_')[1] == 'airplane':
            print('%d%d: %s'%(sample_files.index(f),len(sample_files),f),"\n\n")
            y, sr = librosa.load('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+f+'.wav' ,sr=44100)
            S = librosa.feature.melspectrogram(y=y, sr =sr, n_mels=128)
            sound = librosa.feature.mfcc(S = librosa.power_to_db(S))
            
        
    def extract_stft(part):
        sample_files = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt').read().strip().split('\n')

        for f in sample_files:
    
            #if f.split('_')[1] == 'airplane':
            print('%d%d: %s'%(sample_files.index(f),len(sample_files),f),"\n\n")
            y, sr = librosa.load('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+f+'.wav' ,sr=44100)
            D = librosa.stft(y, n_fft=1024, hop_length=256).T 
            mag, phase = librosa.magphase(D)
            S = np.log(1 + mag * 1000)
            
            np.save('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum_Stft/'+f+'.npy', S)
            
    
    def extract_custom_mfcc(part):
    
        sample_files = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt').read().strip().split('\n')
        min_level_db = -100

        for f in sample_files:
        
            #if f.split('_')[1] == 'airplane':
            print('%d%d: %s'%(sample_files.index(f),len(sample_files),f),"\n\n")
            y, sr = librosa.load('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+f+'.wav')
            mfccs = librosa.feature.mfcc(y=y,sr=44100,n_mfcc=39,n_mels=128,hop_length=int(sr*0.01),n_fft=int(sr*0.02),htk=True) 
            #Norm_S = np.clip((mfccs - min_level_db)/ -min_level_db ,0,1)
            
            np.save('/home/libedev/mute/mute-hero/download/dataset/'+part+'/'+'spectrum_Seogang/'+f+'.npy', mfccs)
            
            
            
            
            
            