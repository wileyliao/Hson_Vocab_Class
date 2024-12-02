import socket
import pyaudio
import struct
import threading
import json

server_address = ('127.0.0.1', 3310)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)  # 連接伺服器

sequence_number = 0

def send_audio_data():
    """
    傳送音訊數據到伺服器
    """
    global sequence_number
    print("開始傳輸音訊數據...")
    try:
        while True:
            data = stream.read(CHUNK)
            # 將序列號與音訊數據組合
            packet = struct.pack("I", sequence_number) + data
            client_socket.sendall(packet)  # 使用 sendall 保證整個數據包被傳輸
            sequence_number += 1
    except Exception as e:
        print(f"傳輸音訊數據時出錯: {e}")
    finally:
        print("停止傳輸音訊數據")
        stop_audio()

def receive_result():
    """
    獨立線程接收伺服器傳回的結果
    """
    try:
        while True:
            # 接收伺服器回傳的數據 (假設以固定長度的 JSON 字串發送)
            data = client_socket.recv(4096)  # 假設每次接收 4096 bytes
            if data:
                result = json.loads(data.decode('utf-8'))
                formatted_result = json.dumps(result, ensure_ascii=False, indent=4)
                print(f"接收到伺服器回傳結果: {formatted_result}")
            else:
                print("伺服器已斷開連線")
                break
    except Exception as e:
        print(f"接收伺服器結果時出錯: {e}")

def stop_audio():
    """
    停止音訊流並關閉 socket
    """
    stream.stop_stream()
    stream.close()
    audio.terminate()
    client_socket.close()


# 啟動發送線程
send_thread = threading.Thread(target=send_audio_data)
send_thread.start()

# 啟動接收線程
receive_thread = threading.Thread(target=receive_result, daemon=True)  # 使用 daemon 線程自動退出
receive_thread.start()

try:
    send_thread.join()  # 等待發送線程執行完畢
except KeyboardInterrupt:
    print("客戶端關閉中...")
    stop_audio()