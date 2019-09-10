import glob
import librosa
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from rxgp1 import mute_constants


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(filenames, file_groups):
    rows = len(filenames)
    features, labels, groups = np.zeros((rows, 193)), np.zeros((rows, 10)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            print("init file : " + fn)
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            y_col = file_groups.index(fn.split('/')[8])
            group = file_groups.index(fn.split('/')[8])
        except Exception as e:
            print("exception:  "+fn)
            print(e)
        else:
            features[i] = ext_features
            labels[i, y_col] = 1
            print("label : " + str(y_col))
            groups[i] = group
            i += 1
    return features, labels, groups


if __name__ == '__main__':
    file_groups = os.listdir(mute_constants.BASE_PATH + "data/train/wav")
    print(file_groups)
    audio_files = []
    for i in range(len(file_groups)):
        audio_files.append(glob.glob(mute_constants.BASE_PATH + "data/train/wav/"+file_groups[i]+"/*.wav"))

    print(len(audio_files))
    for i in range(len(audio_files)):
        files = audio_files[i]
        file_name = file_groups[i]
        X, y, groups = parse_audio_files(files, file_groups)
        for r in y:
            if np.sum(r) > 1.5:
                print('error occured')
        np.savez(mute_constants.BASE_PATH + "data/train/each_npz/urban_sound_%s" % file_name, X=X, y=y, groups=groups)







