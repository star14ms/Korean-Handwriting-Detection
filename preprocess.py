from data import HWKoDataset
from plot import set_font

import torch
import matplotlib.pyplot as plt 
import random
import os

from rich import print
from rich.console import Console
from rich.progress import track
from rich.traceback import install
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

console = Console()


def separate_by_space(x, min_brightness=3, min_space=50, min_letter_len=5):
    '''
    Parameters
    ----------
        ``min_brightness`` ``(float)`` : 
            글자의 감지 여부를 판단할 최소 세로 한 줄 밝기 합
            세로 한 줄 밝기: 세로 한줄의 0~1 범위 픽셀을 ``sum()`` 한 결과
        ``min_space`` ``(int)`` : 
            띄어쓰기로 판단할 최소 빈칸 길이
        ``min_letter_len`` ``(int)`` :
            글자의 감지 여부를 판단할 최소 글자 너비
    '''
    
    row_len = x.shape[2] # (C, H, (W))
    col_len = x.shape[1] # (C, (H), W)

    sep_idxs = []
    detected = False
    appended = False
    space = 0
    xl = 0
    xr = 0

    for now_x in range(1, row_len, 1):
        v_line = x[:, :, now_x:now_x+1]

        if torch.sum(v_line) > min_brightness: # 글씨가 감지되면
            if not detected and appended: # 새로운 감지가 시작되는 순간
                appended = False
                xl = now_x
            space = 0
            detected = True

        else: # 글씨가 감지되지 않으면
            space += 1
            if detected: # 감지가 종료되는 순간
                xr = now_x
            if space == min_space: # 공백 수가 띄어쓰기 너비를 달성하면
                if xr-xl > min_letter_len: 
                    sep_idxs.append((0, col_len, xl, xr))
                appended = True
            detected = False

    if detected and not appended and xr-xl > min_letter_len: # 마지막 감지가 종료되지 않았다면
        sep_idxs.append((0, col_len, xl, row_len))

    return (*sep_idxs,)


def plot_cutting_info(x, letter_idxs, block=True):
    fig = plt.figure()
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
    n_col = 3
    n_row = len(letter_idxs)//n_col+1
    cmap = 'gray' # plt.cm.gray_r

    axes = fig.add_subplot(n_col, 1, 1, xticks=[], yticks=[])
    axes.imshow(x[0], cmap=cmap)
    axes.text(0, -50, f'분해한 조각 수: {len(letter_idxs)}', size=50)
    
    for idx, (yt, yb, xl, xr) in enumerate(letter_idxs):
        x_piece = x[:, yt:yb, xl:xr]
        wh_rate = (xr-xl) / (yb-yt)
        axes = fig.add_subplot(n_col+1, n_row, n_row+idx+1, xticks=[], yticks=[])
        axes.text(0, -20, f'w/h:{wh_rate:.2f} w:{xr-xl}px', size=20, ha="left", color='red' if wh_rate < 0.33 else 'black')
        axes.imshow(x_piece[0], cmap=cmap)

    fig.set_size_inches(12.8, 7.6)
    plt.show(block=block)


def crop_blank_piece(x_piece, sep_idx, min_brightness=3):
    base_yt, _, base_xl, _ = sep_idx
    row_len = x_piece.shape[2] # (C, H, (W))
    col_len = x_piece.shape[1] # (C, (H), W)

    yt = 0
    while torch.sum(x_piece[:, yt:yt+1, :]) < min_brightness and yt < col_len:
        yt += 1

    yb = col_len
    while torch.sum(x_piece[:, yb:yb+1, :]) < min_brightness and yb >= 0:
        yb -= 1

    xl = 0
    while torch.sum(x_piece[:, :, xl:xl+1]) < min_brightness and xl < row_len:
        xl += 1

    xr = row_len
    while torch.sum(x_piece[:, :, xr:xr+1]) < min_brightness and xr >= 0:
        xr -= 1

    return (base_yt + yt, base_yt + yb, base_xl + xl, base_xl + xr)


def crop_blank(x, sep_idxs):
    croping_idxs = []
    for sep_idx in sep_idxs:
        yt, yb, xl, xr = sep_idx
        x_piece = x[:, yt:yb, xl:xr]
        croping_idx = crop_blank_piece(x_piece, sep_idx)
        croping_idxs.append(croping_idx)
    
    return croping_idxs


train_set = HWKoDataset()


for x, t in track(train_set, total=len(train_set)):
    x, t = random.choice(train_set)
    sep_idxs = separate_by_space(x)

    # sep_idxs_no_merge = crop_by_separating_letter(x, recombination=False)
    # croped_idxs2 = crop_blank(sep_idxs)

    print(t)
    plot_cutting_info(x, sep_idxs, block=True)
    # plot_cutting_info(x, sep_idxs_no_merge, block=True)
    # input()

# print(croped_idxs2)
