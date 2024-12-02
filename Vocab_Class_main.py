import time

import torch
import torchaudio
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)

from sklearn.preprocessing import LabelEncoder



label_mapping = {
    "label1": "Crestor",
    "label2": "Diovan",
    "label3": "Lipitor",
    "label4": "Olmetec",
    "label5": "Plavix",
    "label6": "Sevikar",
    "label7": "Twynsta",
}


class AudioClassifier:
    def __init__(self, model_path, label_encoder):
        """
        初始化音訊分類器

        Args:
            model_path (str): 模型儲存路徑
            label_encoder (LabelEncoder): 之前訓練時使用的 LabelEncoder
        """
        # 載入特徵提取器
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            'facebook/wav2vec2-base-960h',
            do_normalize=True
        )

        # 載入模型
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

        # 設置為評估模式
        self.model.eval()

        # 儲存標籤編碼器
        self.label_encoder = label_encoder

    def preprocess_audio(self, audio_path):
        """
        預處理音訊檔案

        Args:
            audio_path (str): 音訊檔案路徑

        Returns:
            torch.Tensor: 處理後的音訊張量
        """
        # 載入音訊
        waveform, sampling_rate = torchaudio.load(audio_path)

        # 如果採樣率不是16kHz，重新取樣
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            waveform = resampler(waveform)
            sampling_rate = 16000

        # 確保一維波形
        waveform = waveform.squeeze()

        # 使用特徵提取器處理
        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=sampling_rate,
            return_tensors='pt',
            padding=True
        )

        return inputs.input_values

    def predict(self, audio_path):
        """
        預測音訊類別

        Args:
            audio_path (str): 音訊檔案路徑

        Returns:
            dict: 包含預測類別和機率的字典
        """
        # 預處理音訊
        inputs = self.preprocess_audio(audio_path)

        # 關閉梯度計算
        with torch.no_grad():
            # 模型推理
            outputs = self.model(inputs)

            # 獲取 logits
            logits = outputs.logits

            # 轉換為概率
            probabilities = torch.softmax(logits, dim=-1)

            # 獲取最可能的類別
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

            # 解碼類別標籤
            predicted_label = self.label_encoder.classes_[predicted_class_idx]

            return {
                'label': predicted_label,
                'probability': probabilities[0][predicted_class_idx].item(),
                'all_probabilities': {
                    self.label_encoder.classes_[i]: prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
            }


# 重新創建 LabelEncoder（需要與訓練時使用相同的類別順序）
label_encoder = LabelEncoder()
label_encoder.fit(["label1", "label2", "label3", "label4", "label5", "label6", "label7"])

# 初始化分類器
classifier = AudioClassifier(
    model_path='./fine_tuned_wav2vec2',
    label_encoder=label_encoder
)

def main(audio_path):
    start_time = time.time()
    # 進行預測
    result = classifier.predict(audio_path)

    # 印出結果
    # print(f"您剛剛念的單字為: {label_mapping.get(result['label'])}")
    # print(f"Prediction Probability: {result['probability']:.4f}")
    # print("\nAll Class Probabilities:")

    results_json = "; ".join([f"{label_mapping.get(label)}, {prob:.4f}" for label, prob in result['all_probabilities'].items()])

    end_time = time.time()

    data_dict = {
        "Data": [
            {
                "name": label_mapping.get(result['label']),
                "conf": f"{result['probability']:.4f}"
            }
        ],
        "Result": results_json,
        "TimeTaken": f"{end_time - start_time:.2f} 秒"
    }

    # print(data_dict)

    return data_dict

if __name__ == "__main__":
    # 測試音訊路徑
    test_audio_path = r"C:\python\torch\voice_print\pill_audio_data\test_data\twynsta_test.wav"
    main(test_audio_path)