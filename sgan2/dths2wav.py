import os
import subprocess
from pathlib import Path

def convert_raw_to_wav(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    # 対象拡張子
    targets = [".dt", ".hs"]

    for src_path in input_root.rglob("*"):
        if src_path.suffix.lower() in targets:
            # 出力先パス（ディレクトリ構造を維持）
            relative_path = src_path.relative_to(input_root)
            dst_path = output_root / relative_path.with_suffix(".wav")

            # 出力ディレクトリ作成
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # sox コマンド実行（raw → wav）
            cmd = [
                "sox",
                "-t", "raw",
                "-r", "16000",     # サンプリングレート
                "-e", "signed",    # 符号付きPCM
                "-b", "16",        # 16bit
                "-c", "1",         # mono
                str(src_path),     # 入力
                str(dst_path)      # 出力WAV
            ]

            print(f"Converting: {src_path} -> {dst_path}")
            subprocess.run(cmd, check=True)

    print("変換完了")

if __name__ == "__main__":
    input_dir = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/RECOG"   # ← 変換元ディレクトリを指定
    output_dir = "/home/sys/s276225/PyTorch-GAN/implementations/sgan2/S-JNAS/RECOG_WAV" # ← 出力先ディレクトリを指定
    
    convert_raw_to_wav(input_dir, output_dir)
