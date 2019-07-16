# -*- coding: utf-8 -*-
# (한글을 쓰려면 위의 주석이 반드시 필요하다.)

from source import mute_constants, logger
import csv


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

    def is_airplane_sound(self):
        f = self.get_file()
        # TODO : 비행기 소음 Boolean return
        return True

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
