import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--labeleddata_npypath", type=str, default="/home/sys/s276225/uninum_hikaru_vector_agevc_RaAlaEars_150_128.npy")
    parser.add_argument("--labeleddata_npypath", type=str, default="/home/s2310062/uninum_kouki_vector_agevc_500e_label_128.npy")
    parser.add_argument("--label_csvpath", type=str, default="/home/s2310062/split_utt_age_new2_school_id_reshape_senior_8class.csv")
    # parser.add_argument("--labellessdata_npypath", type=str, default="/home/sys/s276225/uninum_hikaru_vector_agevc_150e_cv21t_ciairrnd3_pseudo_128.npy")
    # parser.add_argument("--labellessdata_npypath", type=str, default="/home/sys/s276225/uninum_hikaru_vector_agevc_500e_cv21t_ciairrnd3_pseudo_128.npy")
    parser.add_argument("--labellessdata_npypath", type=str, default="/home/s2310062/uninum_kouki_vector_agevc_500e_labelless_takemaru_5k_128.npy")
    parser.add_argument("--enable_ssl", type=bool, default=True) # 半教師あり学習を有効化するか (ラベルなしデータを使うか)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=800) #800
    parser.add_argument("--batch_size", type=int, default=20) #20
    parser.add_argument("--g_lr", type=float, default=2e-4) # 学習率 1e-4
    parser.add_argument("--d_lr", type=float, default=1e-4) # 5e-5
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--label_real", type=float, default=0.9) # ラベルスムージングの適用
    parser.add_argument("--label_fake", type=float, default=0.1)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample_interval", type=int, default=400)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    print(parser.parse_args())
    return parser.parse_args()