import pandas as pd

def merge_wav_and_speaker_info(wav_csv, speaker_csv, output_csv):
    # CSVの読み込み
    wav_df = pd.read_csv(wav_csv)
    speaker_df = pd.read_csv(speaker_csv)

    # speaker_id をキーに結合（左外部結合）
    merged = wav_df.merge(speaker_df, on="speaker_id", how="left")

    # 保存
    merged.to_csv(output_csv, index=False)
    print(f"Saved merged CSV to {output_csv}")


# 使い方
if __name__ == "__main__":
    merge_wav_and_speaker_info(
        wav_csv="wav_info.csv",
        speaker_csv="DOCS/speaker_info.csv",
        output_csv="meta.csv"
    )
