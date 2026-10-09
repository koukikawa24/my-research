import os
import csv

def get_wav_paths(directory):
    wav_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_paths.append(os.path.join(root, file))
    return wav_paths

def read_age_labels(csv_file):
    age_labels = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 1行目をスキップ
        for row in reader:
            key = row[0]  # 1列目
            age = row[4]  # 5列目に年齢ラベルがある
            # print(key)
            age_labels[key] = age
    return age_labels

def save_to_txt(wav_paths, age_labels, output_file):
    with open(output_file, 'w') as f:
        for path in wav_paths:
            parts = path.split('/')
            key = parts[11]  # 0番目から数えてます
            age = age_labels.get(key, 'unknown')
            print(key)
            f.write(f"{os.path.abspath(path)} {age}\n")

wav_directory = '/home/s246245/s226059/workspace/data/age_training/wav/train/ears_train/WAVES'
csv_file = '/home/s246245/s226059/workspace/self_sinctdnn/label_ears.csv'
output_file = '/home/s246245/ears_ken.txt'

wav_paths = get_wav_paths(wav_directory)
age_labels = read_age_labels(csv_file)
save_to_txt(wav_paths, age_labels, output_file)