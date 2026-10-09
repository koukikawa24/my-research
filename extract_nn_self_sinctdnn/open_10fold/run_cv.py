import os
import torch
import gc
from trainAgevector_hikaru_open_10fold import main 

def run_cross_validation():
    # 基本となるパスの設定
    base_train_list_dir = "/home/sys/s276225/extract_nn/data/age_training/wav/cv10_random"
    base_save_dir = "/home/sys/s276225/extract_nn/self_sinctdnn/open_10fold/save_model"
    
    # Fold 01 から 10 までループ
    for fold in range(1, 11):
        fold_str = f"{fold:02d}"  # 1 -> "01", 2 -> "02" とゼロ埋め
        
        print(f"\n" + "="*50)
        print(f" Starting Training: Fold {fold_str} ")
        print("="*50 + "\n")
        
        # 1. train_list のパスを生成
        # /.../cv10_random/fold01/train.train となるように設定
        current_train_list = os.path.join(base_train_list_dir, f"fold{fold_str}", "train.train")
        
        # 2. 保存先のパスを生成（Foldごとにフォルダを分ける）
        current_save_path = os.path.join(base_save_dir, f"agevec_hikaru_fold{fold_str}")
        
        # 3. main関数を呼び出して学習開始
        try:
            main(train_list=current_train_list, save_path=current_save_path)
        except Exception as e:
            print(f"Error occurred in Fold {fold_str}: {e}")
            # エラーが出ても次のfoldに進む場合は continue、止めたい場合は break
            break 
        
        # 4. メモリの解放（各fold終了時にGPUメモリをクリア）
        # 同一プロセスで連続実行する場合、メモリが蓄積されるのを防ぐ
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    run_cross_validation()