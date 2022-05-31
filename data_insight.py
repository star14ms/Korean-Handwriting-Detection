from data import KoHWSentenceDataset
from utils.plot import set_font

import torch
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

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


def plot_sentence_cutting_info(x, letter_idxs, t=None, block=True):
    fig = plt.figure()
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
    n_col = 3
    n_row = len(letter_idxs)//n_col+1
    cmap = 'gray' # plt.cm.gray_r

    # 전체 이미지
    ax = fig.add_subplot(n_col, 1, 1, xticks=[], yticks=[])
    ax.imshow(x[0], cmap=cmap)

    # 이미지 조각 정보 요약
    text = f'분해한 수: {len(letter_idxs)}'
    if t is not None:
        text += f' | 어절 수: {len(t.split())} | 문자 수: {len(t)} | (match, w/h, w)'
    ax.text(0, -50, text, size=40)
    
    # 각 이미지 조각들
    for idx, (yt, yb, xl, xr) in enumerate(letter_idxs):
        x_piece = x[:, yt:yb, xl:xr]
        wh_rate = (xr-xl) / (yb-yt)
        info = f'{wh_rate:.2f}, {xr-xl}'
        ax = fig.add_subplot(n_col+1, n_row, n_row+idx+1, xticks=[], yticks=[])
        ax.text(0, -20, info, size=20, ha="left", color='red' if wh_rate < 0.33 else 'black')
        ax.imshow(x_piece[0], cmap=cmap)

    fig.set_size_inches(12.8, 7.6)
    plt.get_current_fig_manager().window.showMaximized()
    plt.show(block=block)


def plot_brightness_gradient(x, brightness_list, title='', ylim=None, block=True):
    xlim = 0, x.shape[2]
    ylim = 0, max(brightness_list) if ylim is None else ylim
    xs, ys = np.arange(len(brightness_list)), brightness_list

    fig = plt.figure()
    fig.subplots_adjust(top=0.86, bottom=0.105, left=0.068, right=0.988, hspace=0.134, wspace=0.2)
    
    plt.rcParams["font.size"] = 15
    ax = fig.add_subplot(111)
    ax.set_title(title, size=40)
    ax.set_xlabel('image width', size=40)
    ax.set_ylabel('brightness', size=40)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.plot(xs, ys)
    ax.fill_between(xs, ys) # , 0 if ylim is None else ylim

    ax = fig.add_subplot(4, 1, 4, xticks=[], yticks=[]) # , xticks=[], yticks=[]
    ax.imshow(x[0])
    # ax.sharex(axes1)
    
    plt.get_current_fig_manager().window.showMaximized()
    # plt.tight_layout()
    plt.show(block=block)


if __name__ == '__main__':
    train_set = KoHWSentenceDataset()
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
    plt.savefig('insight/Distribution_of_spacing_lengths_{}_to_{}.png'.format(*binrange))
    plt.show()