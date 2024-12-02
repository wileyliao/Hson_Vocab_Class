import socket
import json
import threading

import numpy as np
import pyaudio
import wave
from Vocab_Class_main import main

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class AudioServerTCP:
    def __init__(self, host='0.0.0.0', port=3310):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.CHUNK = 1024
        print("TCP 伺服器已啟動，等待連接...")


    def send_result(self, connection, result):
        """
        傳送結果給連線的客戶端
        """
        try:
            # 將字典轉換為 JSON 字串後發送
            message = json.dumps(result)
            connection.sendall(message.encode('utf-8'))
            print("結果已成功發送！")
        except Exception as e:
            print(f"發送結果時發生錯誤: {e}")


    def start(self):
        connection, addr = self.server_socket.accept()
        print(f"客戶端已連接: {addr}")

        # 初始化變數
        dataset = []  # 存儲所有數據
        high_energy_queue = []  # 累積高能量片段
        low_energy_count = 0  # 記錄低能量次數
        finalize_dataset = False  # 是否進行儲存標記
        accumulate_extra_count = 0  # 額外累積的片段計數

        while True:
            try:
                data = connection.recv(4 + self.CHUNK * 2)  # 假設最多接收 int16 格式
                if not data:
                    break

                audio_samples = np.frombuffer(data[4:], dtype=np.int16)
                dataset.append(audio_samples)

                abs_samples = np.abs(audio_samples)
                abs_avg = np.average(abs_samples)

                if abs_avg >= 1000:
                    high_energy_queue.append(audio_samples)
                    # print(f"錄製中, 高能量累積: {len(high_energy_queue)}, 平均值: {abs_avg}")
                    low_energy_count = 0  # 重置低能量樣本計數

                else:
                    low_energy_count += 1
                    # print(f"無錄製, 低能量累積: {low_energy_count}, 平均值: {abs_avg}")

                if low_energy_count == 10:
                    if len(high_energy_queue) >= 5:
                        finalize_dataset = True
                        accumulate_extra_count = 0
                        # print("高能量條件達成，準備額外累計")
                    else: # 高能量不足
                        dataset.clear()
                        high_energy_queue.clear()
                        low_energy_count = 0
                        # print("高能量不足，清空數據，重新開始")

                # 額外累積檢查
                if finalize_dataset:
                    accumulate_extra_count += 1
                    if accumulate_extra_count >= 5:  # 額外累積完成
                        # 將 dataset 儲存為檔案
                        print("判斷中....")
                        combined_samples = np.concatenate(dataset)
                        with wave.open("vocab_test.wav", "wb") as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(combined_samples.tobytes())

                        result = main("vocab_test.wav")
                        print("判斷結果： ", result.get('Data')[0]['name'])
                        print("信心閥值： ", result.get('Data')[0]['conf'])
                        result_thread = threading.Thread(target=self.send_result, args=(connection, result))
                        result_thread.start()

                        # print("高能量音頻已保存為 high_energy_recording.wav")
                        # 清空所有數據，重置狀態
                        dataset.clear()
                        high_energy_queue.clear()
                        finalize_dataset = False
                        low_energy_count = 0

            except Exception as e:
                print(f"接收數據錯誤: {e}")
                break

        connection.close()
        print("客戶端連接已關閉")


if __name__ == "__main__":
    server = AudioServerTCP()
    server.start()

