import os, sys
sys.path.insert(0, os.path.abspath(".."))
from flask import Flask, json, render_template, send_from_directory, request
from rxgp1.opensky_api import OpenSkyApi
import time
from rxgp1 import logger, file_util, mute_constants
import sys
import shutil

# 플래스크 앱 생성
app = Flask(__name__, static_url_path='', static_folder='resources', template_folder='html')

app.config['DEBUG']=True

# 최대 업로드 설정a
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 10

# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = mute_constants.UPLOAD_PATH


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
                if file_obj.is_exists_file():
                    logger.debug('file is exists')
                    if not (os.path.isdir('/home/libedev/mute/mute-hero/air_save/' + str(name))):
                        os.makedirs(os.path.join('/home/libedev/mute/mute-hero/air_save/' + str(name)))

                    destination_path = '/home/libedev/mute/mute-hero/air_save/' + str(name) + "/"
                    shutil.copy2(audio_file_path, destination_path)

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
        if not s.on_ground:
            lst.append(obj)
    response = app.response_class(
        response=json.dumps(lst),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)