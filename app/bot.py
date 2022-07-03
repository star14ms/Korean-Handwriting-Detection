import threading
import json
import socket
import os
import PIL.Image as Image

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch import tensor, load, float32
from kohwctop.model import KoCtoP, ConvNet
from kohwctop.transform import Resize
from kohwctop.test import predict
from utils.rich import console


def to_client(conn, addr):
    try:
        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        console.log('===========================')
        console.log('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            console.log('클라이언트 연결 끊어짐')
            exit(0)

        recv_json_data = json.loads(read.decode())
        img_path = recv_json_data['img_path']

        img = Image.open(img_path).convert('L')
        answer = detect(img)

        response = {
            "answer": answer,
        }
        
        console.log('예측:', answer)
        message = json.dumps(response)
        conn.send(message.encode())

    except Exception as ex:
        console.log(ex)

    finally:
        conn.close()


def detect(img):
    model = ConvNet()
    model_CtoP = KoCtoP()
    model.load_state_dict(load('model.pt'))
    model_CtoP.load_state_dict(load('model_CtoP.pt'))

    resize = Resize()
    img = resize(img)
    answer = predict(img, None, model, model_CtoP)

    return answer


class BotServer:
    def __init__(self, srv_port, listen_num):
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None

    # sock 생성
    def create_sock(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mySock.bind(("0.0.0.0", int(self.port)))
        self.mySock.listen(int(self.listen))
        return self.mySock

    # client 대기
    def ready_for_client(self):
        return self.mySock.accept()

    # sock 반환
    def get_sock(self):
        return self.mySock


if __name__ == '__main__':

    port = 5050
    listen = 32

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    console.log("bot start")

    while True:
        conn, addr = bot.ready_for_client()

        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
        ))
        client.start()