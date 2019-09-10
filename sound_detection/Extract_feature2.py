import glob
import librosa
import numpy as np

def extract_feature(file_name):
    
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(filenames):
    
    rows = len(filenames)
    features, labels, groups = np.zeros((rows,193)), np.zeros((rows,11)), np.zeros((rows, 1))
    i = 0
    
    for fn in filenames:
        
        try:
            
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('/')[10].split('-')[1])
            group = int(fn.split('/')[10].split('-')[0])
        
        except:
            
            print(fn)
            
        else:
            
            features[i] = ext_features
            labels[i, y_col] = 1
            groups[i] = group
            i += 1
            
    return features, labels, groups


audio_files =[]

for i in range(1,12):
    
    audio_files.extend(glob.glob('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/fold%d/*.wav'%i))

#print(audio_files[0:100])
#feature, labels, groups = parse_audio_files(audio_files[0:100])
#print(np.array(feature).shape,feature,"\n\n",np.array(labels).shape,labels,"\n\n",np.array(groups).shape,groups,"\n\n")



for i in range(9):
    files = audio_files[i*1000: (i+1)*1000]
    print(files)
    X,y, groups = parse_audio_files(files)
    print("X >>",X, "\n\n","y >> ", y ,"\n\n","groups >> ", groups,"\n\n")
        
    
    print(files)
    X,y, groups = parse_audio_files(files)
    
    for r in y:
        if np.sum(r) > 1.5:
            print('error occured')
            break
    np.savez('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/f_DATASET_%d.npz'%i,X=X,y=y,groups=groups)