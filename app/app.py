import os
import json
import re
import time
import socket
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import urllib.request
from cairosvg import svg2png


UPLOAD_FOLDER = 'app/image/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'svg'}

# 서버 접속 정보
host = "127.0.0.1"  # 서버 IP 주소
port = 5050  # 서버 통신 포트

# Flask 어플리케이션
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, resources={'/detect': {'origins': '*' }}, )


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_path(filename=None):
    date = time.strftime('%y%m%d', time.localtime())
    save_dir = app.config['UPLOAD_FOLDER']+date
    os.makedirs(save_dir, exist_ok=True)

    if filename is None: 
        filename = str(int(time.time())) + '.png'
    filename = secure_filename(filename)
    assert allowed_file(filename)
    filepath = os.path.join(save_dir, filename)
    
    return filepath


# 서버와 통신
def _detect(img_path):
    # 서버 연결
    mySocket = socket.socket()
    mySocket.connect((host, port))

    json_data = {
        'img_path': img_path,
    }
    message = json.dumps(json_data)
    mySocket.send(message.encode())

    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)

    mySocket.close()

    return ret_data


@app.route('/', methods=['GET'])
def index():
    return 'hello', 200


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            data = request.data.decode('utf-8')
            response = urllib.request.urlopen(data)
            svg_code = response.file.read()
            filepath = get_file_path()
            svg2png(bytestring=svg_code, write_to=filepath)
        else:
            file = request.files['file']
            filepath = get_file_path(file.filename)
            file.save(filepath)
            
        ret = _detect(filepath)
        return jsonify(ret)
    
    except Exception as e:
        # 오류 발생시 500 오류
        print(repr(e))
        abort(500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
