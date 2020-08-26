import os, sys
sys.path.insert(0, os.path.abspath(".."))
from flask import Flask, json, render_template, send_from_directory, request
from rxgp1.opensky_api import OpenSkyApi
import time
from rxgp1 import logger, mute_constants, send_to_db, file_util
import sys
import shutil
import librosa
import numpy as np



# 판별 시스템에 들어갈 패키지들 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import random
import os
import PYCLASS2_air_Copy1
from torch.utils.data import Dataset, DataLoader
import PYCLASS2_air_test
import CNN_CLASS
import librosa
import CNN_CLASS2

import time

# 플래스크 앱 생성
app = Flask(__name__, static_url_path='', static_folder='resources', template_folder='html')

app.config['DEBUG']=True

# 최대 업로드 설정a
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 10

# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = mute_constants.UPLOAD_PATH   # upload 폴더 바꿨음 2로


@app.route('/resources/js/<path:path>')
def send_js(path):
    return send_from_directory('resources/js/', path)


@app.route('/resources/css/<path:path>')
def send_css(path):
    return send_from_directory('resources/css', path)


@app.route('/resources/images/<path:path>')
def send_images(path):
    return send_from_directory('resources/images/', path)


@app.route("/map")
def hello():
    return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        name = request.form['name']
        extension = os.path.splitext(str(file.filename))[1]
        if file:
            if extension == '.wav':
                milli_sec = int(round(time.time() * 1000))
                file_name = 'Sound_' + str(milli_sec) + '_' + file.filename
                logger.debug('[file origin name : ' + file.filename + ']')
                logger.debug('[file name : ' + file_name + ' ]')
                audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                file.save(audio_file_path)
                file_obj = file_util.io(audio_file_path, 'r')
                if file_obj.is_exists_file(): # 이 밑에 구현
                    
                    logger.debug('file is exists')
                    if not (os.path.isdir('/home/libedev/mute/mute-hero/air_save/' + str(name))):
                        os.makedirs(os.path.join('/home/libedev/mute/mute-hero/air_save/' + str(name)))

                    destination_path = '/home/libedev/mute/mute-hero/air_save/' + str(name) + "/"
                    shutil.copy2(audio_file_path, destination_path)
                    # sound decibel
                    y, sr = librosa.load(file_obj.get_path())
                    S = np.abs(librosa.stft(y))
                    sound = librosa.core.amplitude_to_db(S)

                    db_level = round(float(np.max(sound)), 2)

                    send_to_db.send_to_db(file_name, name, db_level)

                    obj = dict(status=0)
                else:
                    obj = dict(status=2)
            else:
                obj = dict(status=2)
        else:
            obj = dict(status=3)
        response = app.response_class(
            response=json.dumps(obj),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        obj = dict(status=500)
        response = app.response_class(
            response=json.dumps(obj),
            status=500,
            mimetype="application/json"
        )
        logger.debug(sys.exc_info)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return response



    
    
    
    
# 판별시스템 함수
    
@app.route('/sound', methods=['POST'])
def upload_file2():
    try:
        file = request.files['file']
        name = request.form['name']
        extension = os.path.splitext(str(file.filename))[1]
        if file:
            
            
            if extension == '.wav':
                
                start = time.time() # 시간 확인 
                
                
                milli_sec = int(round(time.time() * 1000))
                file_name = 'Sound_' + str(milli_sec) + '_' + file.filename
                logger.debug('[file origin name : ' + file.filename + ']')
                logger.debug('[file name : ' + file_name + ' ]')
                audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                file.save(audio_file_path)
                file_obj = file_util.io(audio_file_path, 'r')
                print(audio_file_path)
                
                
                if file_obj.is_exists_file(): # 이 밑에 구현
                    
                    logger.debug('file is exists')
                    
                    # audio_file_path로 음원읽어드림. '/home/libedev/mute/mute-hero/air_save/test/'
                    
                    X, sample_rate = librosa.load(audio_file_path)   
                    
                    
                    
                    stft = np.abs(librosa.stft(X))
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
                    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
                    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)       
                    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                    
                    logger.debug("stage 1")
                    
                    # 딥러닝용 데이터로 변환
                    
                    s_data = ext_features
                    train_data2 = s_data
                    x=np.array([0,0,0,0,0,0,0])
                    train_data = np.zeros((1,200))
                    train_data =  np.concatenate((train_data2,x),axis=None)
                    train_data = train_data.reshape(1,1,10,20)
                    dtype = torch.FloatTensor
                    
                    logger.debug("stage 2")
                    # pytorch 용 데이터프레임으로 변환됨 ---> train_data (완성)
                    
                    train_data  = torch.as_tensor(train_data).type(dtype)
                    
                    logger.debug("stage 3")
                    # 미리 학습된 모델 불러오기
                    
                    save_path = '/home/libedev/mute/mute-hero/download/dataset/model2/'
                    PATH  = save_path + 'model3.pkl'
                    model = CNN_CLASS2.CNN2()
                    model.load_state_dict(torch.load(PATH))
                    
                    logger.debug("stage 4")
    
                    # 딥러닝용으로 변환한 데이터를 불러온 모델을 통해 판별하기   
    
                    logger.debug("file_name : " + file_name)
                    #result=predict(file_name)
                    out = model(train_data)
                    _, predicted = torch.max(out.data, 1)
                    
                    
                    
                    tm = time.time() - start # 판별할때의 걸린시간
                    
                    
                    
                    logger.debug("stage 5")
                    
                    if not (os.path.isdir('/home/libedev/mute/mute-hero/air_save/' + str(name))):
                        os.makedirs(os.path.join('/home/libedev/mute/mute-hero/air_save/' + str(name)))

                   

                    if predicted[0] == 2:  # 비행음으로 판별

                        obj = dict(status=0)
                        
                        print("Analysis Time : " ,tm)
                    
                    elif predicted[0] == 1:   # 비행소음이외의 음으로 판별
                    
                        obj = dict(status=2)
                    
                        print("Analysis Time : " ,tm)
            else:
                
                obj = dict(status=2)
        else:
            
            obj = dict(status=3)
            
        response = app.response_class(response=json.dumps(obj),status=200,mimetype='application/json')
        
        #response2 = app.response_class(response2=json.dumps(obj2),status=200,mimetype='application/json')
        
        return response
    
    except Exception as e:
        obj = dict(status=500)
        response = app.response_class(
            response=json.dumps(obj),
            status=500,
            mimetype="application/json"
        )
        logger.debug(sys.exc_info)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
        return response 
    
    
    
    
    
    
@app.route('/airplane', methods=['GET'])
def get_airplane_location():
    api = OpenSkyApi('rxgp1', 'tla0420!@')
    states = api.get_states(bbox=(34.3900458847, 38.6122429469, 126.117397903, 129.468304478))  # In Korea
    lst = []
    for s in states.states:
        obj = {
            'latitude': s.latitude,
            'longitude': s.longitude,
            'callsign': s.callsign,
            'geo_altitude': s.geo_altitude,
            'on_ground': s.on_ground,
            'heading': s.heading
        }
        lst.append(obj)
    response = app.response_class(
        response=json.dumps(lst),
        status=200,
        mimetype='application/json'
    )
    
    return response




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
