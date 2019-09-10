#-*- coding: utf-8 -*-
# (한글을 쓰려면 위의 주석이 반드시 필요하다.)

# 상수모음 클래스

# 프로젝트 경로 (test server)
BASE_PATH = "/home/libedev/mute/mute-hero/"

# 프로젝트 경로 (local)
# BASE_PATH = "/Users/alex/works/workspaces/work_mute/mute-hero/"
UPLOAD_PATH = BASE_PATH + "upload/"
DOWNLOAD_PATH = BASE_PATH + "download/"
TRAIN_PATH = DOWNLOAD_PATH + "train/"
TEST_PATH = DOWNLOAD_PATH + "test/"
TEMP_PATH = DOWNLOAD_PATH + "temp/"
FREESOUND_PATH = DOWNLOAD_PATH + "freesound/"
CSV_PATH = BASE_PATH + "csv/"

# 파일 형식의 로그 경로 (Daily Rolling)
LOG_PATH = BASE_PATH + "logs/"
