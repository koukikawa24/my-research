import os
import csv

ENCODINGS = ["utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp"]

def read_text_with_fallback(path):
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return [line.strip() for line in f.readlines()]
        except Exception:
            continue
    raise UnicodeDecodeError("All encoding attempts failed", path, 0, 0, "Unknown encoding")

def collect_personal_info(personals_dir, output_csv):
    records = []

    for fname in os.listdir(personals_dir):
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(personals_dir, fname)

        try:
            lines = read_text_with_fallback(file_path)

            if len(lines) < 6:
                print(f"Warning: {file_path} の形式が不正です（行数不足）")
                continue

            speaker_id = lines[0]
            age        = lines[4]
            sex        = lines[5]

            records.append([speaker_id, age, sex])

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # CSV保存
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_id", "age", "sex"])
        writer.writerows(records)

    print(f"Saved {len(records)} records to {output_csv}")


# 使い方
if __name__ == "__main__":
    personals_dir = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/MODEL_WAV/DOCS/PERSONALS"
    output_csv = "speaker_info.csv"
    collect_personal_info(personals_dir, output_csv)
