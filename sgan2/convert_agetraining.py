import pandas as pd
import os

# ---- 入力ファイル ----
path_csv = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/MODEL_WAV/s-jnas_sampled_label.csv"   # /home/.../M1F01/...wav,7
info_csv = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/MODEL_WAV/DOCS/speaker_info.csv"    # M1F01,64,F

# ---- CSV読み込み ----
df_path = pd.read_csv(path_csv, header=None, names=["path", "label"])
df_info = pd.read_csv(info_csv, header=None, names=["id", "age", "sex"])

# sex を小文字にしておく
df_info["sex"] = df_info["sex"].str.lower()

# ---- ID → (age, sex) 辞書作成 ----
info_dict = df_info.set_index("id")[["age", "sex"]].to_dict("index")

# ---- 新しい出力の作成 ----
output_rows = []

for p in df_path["path"]:
    # パスから M1F01 のような ID を抽出（末尾の2階層を利用）
    parts = p.split("/")
    spk_id = parts[-2]  # M1F01 など

    if spk_id in info_dict:
        age = info_dict[spk_id]["age"]
        sex = info_dict[spk_id]["sex"]
        output_rows.append(f"{p} {age}{sex}")
    else:
        # ID が見つからなかった場合
        output_rows.append(f"{p} UNKNOWN")

# ---- 出力CSVに保存 ----
with open("converted.csv", "w") as f:
    for line in output_rows:
        f.write(line + "\n")

print("converted.csv を作成しました")
