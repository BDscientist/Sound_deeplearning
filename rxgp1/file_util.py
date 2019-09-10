# -*- coding: utf-8 -*-
# (한글을 쓰려면 위의 주석이 반드시 필요하다.)

import csv
import numpy as np
import tensorflow as tf
import librosa
from rxgp1 import logger, mute_constants



# File 의 io 를 편하게 하기위한 클래스
class io:
    # 기본 프로젝트 경로
    __base_path: str = mute_constants.BASE_PATH
    # 쓰거나 읽어올 파일 경로
    __file_path: str = ""
    # 파일 존재 유무
    __is_exists_file: bool = True
    # 파일 객체
    __file: object = None
    # 파일 처리 옵션
    __option: str = ""

    def __init__(self, file_path, option):
        self.__file_path = file_path
        self.__option = option
        self.__init()

    def __init(self):
        logger.info("read/write file : " + self.__file_path + " , option : " + self.__option)
        try:
            f = open(self.__file_path, self.__option, encoding='utf-8')
            self.__file = f
        except (FileNotFoundError, IOError, ValueError) as error:
            self.__is_exists_file = False
            logger.error(self.__file_path + " is error : " + error)

    def get_path(self):
        return self.__file_path

    def is_exists_file(self):
        return self.__is_exists_file

    def get_file(self):
        if self.is_exists_file():
            if self.__file is not None:
                return self.__file
            else:
                raise FileNotFoundError
        else:
            raise FileNotFoundError

    def close(self):
        if self.is_exists_file():
            if self.__file is not None:
                self.__file.close()
            else:
                raise FileNotFoundError
        else:
            raise FileNotFoundError

    def write(self, param):
        if self.is_exists_file():
            if self.__file is not None:
                self.__file.write(param)
            else:
                raise FileNotFoundError
        else:
            raise FileNotFoundError


class extract_util_for_tensorflow:
    # 쓰거나 읽어올 파일 경로
    __file_path: str = ""
    __sess: None

    __X: None
    __y_sigmoid: None
    __y_: None

    def __init__(self, file_path):
        # 텐서플로우 모델 생성
        self.__file_path = file_path

        n_dim = 193
        n_classes = 10
        n_hidden_units_one = 300
        n_hidden_units_two = 200
        n_hidden_units_three = 100
        sd = 1 / np.sqrt(n_dim)

        self.__X = tf.placeholder(tf.float32, [None, n_dim])
        Y = tf.placeholder(tf.float32, [None, n_classes])

        W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
        b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
        h_1 = tf.nn.sigmoid(tf.matmul(self.__X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
        b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
        h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

        W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
        b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
        h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

        W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
        b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
        z = tf.matmul(h_3, W) + b
        self.__y_sigmoid = tf.nn.sigmoid(z)
        self.__y_ = tf.nn.softmax(z)

        init = tf.global_variables_initializer()

        # 모델 파라메타 로드
        saver = tf.train.Saver()
        self.__sess = tf.Session()
        self.__sess.run(init)
        saver.restore(self.__sess, mute_constants.BASE_PATH + 'data/train/model/model_321.ckpt')

    def extract_feature(self):
        X, sample_rate = librosa.load(self.__file_path)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz

    def is_airplane_sound(self):
        # ['CarEngine', 'airplane', 'Noise', 'Laughter', 'ManSpeech', 'WomanSpeech', 'Conversation', 'Horn', 'Vehicle']
        mfccs, chroma, mel, contrast, tonnetz = self.extract_feature()
        x_data = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        y_hat, sigmoid = self.__sess.run([self.__y_, self.__y_sigmoid], feed_dict={self.__X: x_data.reshape(1, -1)})
        index = np.argmax(y_hat)
        print(sigmoid)
        is_airplane_index = False
        if index == 1: # 상위 인덱스의 비행기 소리 처리
            is_airplane_index = True
        return is_airplane_index


class csv_util:
    # 기본 프로젝트 경로
    __file_path: str = mute_constants.CSV_PATH
    __file: io = None
    __d = {}

    def __init__(self, file_name):
        self.__file_path = self.__file_path + file_name
        self.__file = io(self.__file_path, 'r')
        self.__d["/m/0k4j"] = "CAR"
        self.__d["/m/02mk9"] = "CarEngine"
        self.__d["/m/01h8n0"] = "Conversation"
        self.__d["/m/0912c9"] = "Horn"
        self.__d["/m/0ytgt"] = "kidspeech"
        self.__d["/m/01j3sz"] = "Laughter"
        self.__d["/m/05zppz"] = "ManSpeech"
        self.__d["/m/096m7z"] = "Noise"
        self.__d["/m/0k5j"] = "Plane"
        self.__d["/m/07yv9"] = "Vehicle"
        self.__d["/m/03m9d0z"] = "Wind"
        self.__d["/m/02zsn"] = "WomanSpeech"

    def is_csv_exists(self):
        return self.__file.is_exists_file()

    def get_csv_object(self):
        return csv.reader(self.__file.get_file())

    def close(self):
        self.__file.close()

    def get_category(self, param):
        try:
            return self.__d[param]
        except KeyError:
            return None
