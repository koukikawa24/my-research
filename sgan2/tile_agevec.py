import os
import numpy as np
from tqdm import tqdm

def remove_extension(path):
    base, _ = os.path.splitext(path)
    return base

def tile_vector(path="/home/sys/s276225/uninum_ken_vector_agevc_RaAlaEars_150.npy"):
    npy_str = remove_extension(path)
    npy_path = os.path.join(npy_str+'.npy')
    x = np.load(npy_path)

    x_tile = np.empty((len(x),128,512))
    #xのデータを128＊512にする（引き延ばして画像とする）
    for num,i in tqdm(enumerate(x),total=len(x)):
        i = np.tile(i,(128,1))
        x_tile[num] = i

    print(np.shape(x_tile))
    #save
    save_use_path = npy_str + '_128.npy'
    np.save(save_use_path,x_tile)
    print(f'saved to {save_use_path}')

if __name__ == '__main__':
    path = "/home/s2310062/uninum_kouki_vector_agevc_500e_labelless_takemaru_2.5k.npy"
    tile_vector(path)