from data import HWKoDataset
from plot import set_font

import torch
import seaborn as sns
import matplotlib.pyplot as plt 

from rich.progress import track
import os

set_font(family='BM JUA_TTF')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_spacing_lengths(x, min_brightness=3, min_space=20):
    '''
    Parameters
    ----------
        ``min_brightness`` ```(float)```: 
            글자의 감지 여부를 판단할 최소 세로 한 줄 밝기 합
            세로 한 줄 밝기: 세로 한줄의 0~1 범위 픽셀을 ```sum()``` 한 결과
        ``min_space`` ```(int)```: 
            띄어쓰기로 판단할 최소 빈칸 길이
    '''
    
    row_len = x.shape[2] # (C, H, (W))

    lens_space = []
    space = 0
    detected = False
    appended = False

    for now_x in range(1, row_len, 1):
        v_line = x[:, :, now_x:now_x+1]

        if torch.sum(v_line) > min_brightness: # 글씨가 감지되면
            if not detected and appended: # 새로운 감지가 시작되는 순간
                lens_space.append(space)
                appended = False
            space = 0
            detected = True

        else: # 글씨가 감지되지 않으면
            space += 1
            if space == min_space: # 공백 수가 띄어쓰기 너비를 달성하면
                appended = True
            detected = False

    return (*lens_space,)


if __name__ == '__main__':
    train_set = HWKoDataset()
    specimen_len = len(train_set)
    binrange = (20, 120)

    lens_space_of_dataset = ()
    idx = 0

    for x, t in track(train_set, total=specimen_len):
        lens_space_x = get_spacing_lengths(x)
        lens_space_of_dataset += lens_space_x
        
        idx += 1
        if idx > specimen_len:
            break
    
    sns.histplot(lens_space_of_dataset, binrange=binrange, bins=100, kde=False)
    plt.title('문장 공백 길이 분포도\nDistribution of spacing lengths')
    plt.savefig('data_insight/Distribution_of_spacing_lengths_{}_to_{}.png'.format(*binrange))
    plt.show()