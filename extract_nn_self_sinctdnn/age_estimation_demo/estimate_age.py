import sys
import time
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa
from sinctdnn import SincTDNN

label2class = {
    0 : "未満児",
    1 : "幼稚園児",
    2 : "小学校低学年",
    3 : "小学校中学年",
    4 : "小学校高学年",
    5 : "中学生",
    6 : "高校生",
    7 : "大人",
    8 : "Generator"
}

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),\
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        ds_size_h = 128 // 2 ** 5
        ds_size_w = 512 // 2 ** 5

        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size_h*ds_size_w, 1),\
                                       nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128*ds_size_h*ds_size_w, 8+1),\
                                       nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class PretrainedSincTDNN(nn.Module):
    def __init__(self, model, classes=188):
        super().__init__()
        self.sinctdnn = model
        self.output = nn.Linear(512, classes)

    def forward(self, x):
        x = self.sinctdnn(x)
        x = self.output(x)
        return x

def extract_features(model, target_module, inputs):
    features = None

    def forward_hook(module, inputs, outputs):
        nonlocal features
        features = outputs.detach().clone()

    handle = target_module.register_forward_hook(forward_hook)
    model.eval()
    with torch.no_grad():
        model(inputs)
    handle.remove()

    return features

def extractor(audio):
    # モデルのロード
    model = SincTDNN()

    model = PretrainedSincTDNN(model, classes=188)
    model_path = 'agevc_RaAlaEars_150.model'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    inputs = audio

    # 特徴量の抽出
    target_module = model.sinctdnn.tdnn_.segment7  # 抜きたい中間層
    features = extract_features(model, target_module, inputs)

    return features

# wavファイルを読み込む関数
def load_audio_from_wav(path, fs=16000):
    print(f"Loading WAV file: {path} ...")
    signal, _ = librosa.load(path, sr=fs) # リサンプリングして読み込み
    signal = librosa.util.normalize(signal)
    # ここから確認用
    # print("Play the recording...")
    # sd.play(signal, fs)
    # sd.wait()
    inp = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(2)
    return inp

def notify_recording(duration):
    COUNT_DOWN_SEC = 5
    print(f'Please speak for {np.int(duration)} seconds after the countdown.')
    time.sleep(1)
    for i in range(COUNT_DOWN_SEC, -1, -1):
        # キャリッジリターンっていうらしい。へ～。
        sys.stdout.write(f'\rStarting in {i} seconds.')
        sys.stdout.flush()
        time.sleep(1)
    print()

# 音声を録音する関数
def record_audio(duration=4.0, fs=16000):
    notify_recording(duration)
    print("Recording...")
    recording = sd.rec(int(duration * fs),\
                       samplerate=fs,\
                       channels=1,\
                       dtype='float32')
    sd.wait()
    print("Recording finished.")
    signal = recording[:, 0] # チャンネル次元を削除
    signal = librosa.util.normalize(signal)
    # ここから確認用
    # print("Play the recording...")
    # sd.play(signal, fs)
    # sd.wait()
    inp = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(2)
    return inp

def main():
    # Discriminatorのロード
    print("Loading Discriminator model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load('best_discriminator_fold7.pt',\
                                             map_location=device))
    discriminator.eval()

    # 音声の入力とage-vectorの抽出
    audio = record_audio(duration=4.0, fs=16000)
    #audio = load_audio_from_wav(path=r"test\091208202426.121.103.247.154-3-2m.wav", fs=16000)

    print("Extracting Age-vector...")
    age_vec = extractor(audio)

    # age-vectorを128x512に引き延ばす
    age_vec_tile = age_vec.repeat(128, 1)

    # 音声特徴量をDiscriminatorに入力
    print("Estimating age...")
    input_tensor = age_vec_tile.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, label_probs = discriminator(input_tensor)
        predicted_label = torch.argmax(label_probs, dim=1).item()

    # 結果の出力
    print(f'あなたは{label2class[predicted_label]}ですね！')


if __name__ == '__main__':
    main()