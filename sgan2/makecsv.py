import os
import csv
import wave

def get_wav_duration(wav_path):
    """wavファイルのduration(秒)を返す"""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def collect_wav_info(model_dir, output_csv):
    target_dirs = ["WAVES_DT", "WAVES_HS"]
    records = []

    for tdir in target_dirs:
        dir_path = os.path.join(model_dir, tdir)
        if not os.path.isdir(dir_path):
            print(f"Warning: {dir_path} が存在しません")
            continue

        # 話者ごとのディレクトリを探索
        for speaker_id in os.listdir(dir_path):
            speaker_path = os.path.join(dir_path, speaker_id)

            if not os.path.isdir(speaker_path):
                continue

            # wavファイルを探索
            for fname in os.listdir(speaker_path):
                if fname.lower().endswith(".wav"):
                    wav_path = os.path.abspath(os.path.join(speaker_path, fname))

                    try:
                        duration = get_wav_duration(wav_path)
                    except Exception as e:
                        print(f"Error reading {wav_path}: {e}")
                        continue

                    records.append([wav_path, speaker_id, duration])

    # CSVとして保存
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "speaker_id", "duration(sec)"])
        writer.writerows(records)

    print(f"Saved {len(records)} records to {output_csv}")


# 使い方
if __name__ == "__main__":
    model_dir = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/MODEL_WAV"     # 例: "/home/sys/s276225/MODEL_WAV"
    output_csv = "wav_info.csv"
    collect_wav_info(model_dir, output_csv)
