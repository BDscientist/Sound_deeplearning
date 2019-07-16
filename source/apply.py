import sys
import os
import io
import ML2
import numpy as np
import prediction
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


sound_data = pd.read_csv('c:/project2/test_data.csv',skiprows=[0],header=None)
#print(sound_data)

#model_ovr = OneVsRestClassifier(LogisticRegression()).fit(sound_data[,])

f_sound_data = sound_data.loc[:,"0":"249"]
f_sound_target = sound_data.loc[:,"250":"250"]

print(f_sound_data)
print("\n\n\n\n")
print(f_sound_target)

#데이터 파악
#sns.pairplot(f_sound_data , hue=f_sound_target)
#plt.show()

seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

dataset = sound_data.values
X = f_sound_data.astype(float)
Y_obj = f_sound_target

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim =250 ,activation='relu' ))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
#model 실행
model.fit(X,Y_encoded, epochs=5000,batch_size=1)

print("\n Accuracy: %.4f "%(model.evaluate(X,Y_encoded)[1]))
