import pandas as pd

def sample_by_speaker(input_csv, output_csv, n_samples=5, duration_threshold=4.0, seed=42):
    # CSV読み込み
    df = pd.read_csv(input_csv)

    # duration(sec) >= threshold のものだけ
    filtered = df[df["duration(sec)"] >= duration_threshold]

    # 話者IDごとにサンプリング
    sampled_list = []

    for speaker, group in filtered.groupby("speaker_id"):
        # 話者ごとに、サンプリング可能かチェック
        if len(group) >= n_samples:
            sample = group.sample(n=n_samples, random_state=seed)
            sampled_list.append(sample)
        else:
            print(f"Warning: speaker {speaker} has only {len(group)} rows (< {n_samples}), skipping.")

    # 結合
    if sampled_list:
        final_df = pd.concat(sampled_list, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Saved {len(final_df)} rows to {output_csv}")
    else:
        print("No speakers met the sampling requirement.")


# 使い方例
sample_by_speaker(
    input_csv="meta.csv",
    output_csv="sampled.csv",
    n_samples=3,              # 各話者から取りたい数
    duration_threshold=4.0
)
