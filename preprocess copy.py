import matplotlib.pyplot as plt 
import torch

from rich import print
from rich.console import Console
from rich.traceback import install
from rich.progress import track

from data import HWKoDataset
from plot import set_font
from utils import bcolors as bc
install()
set_font(family='BM JUA_TTF')
console = Console()


def crop_by_separating_letter(x, min_brightness=3, wh_rate_range=(0.4, 1.1), min_space=30, recombination=True):
    '''
    Parameters
    ----------
        ``min_brightness`` (float): 
            한 글자로 인정할 최소 세로 한 줄 밝기
            세로 한줄의 (0~1)픽셀을 sum() 한 결과
        ``max_wh_rate`` (float): 
            한 글자로 인정할 최대 세로/가로 비율
    '''
    row_len = x.shape[2] # (C, H, (W))
    col_len = x.shape[1] # (C, (H), W)
    wh_rate_min = wh_rate_range[0]
    wh_rate_max = wh_rate_range[1]

    sep_idxs = []
    sep_idxs_temp = []
    l = None
    before_r = None
    before_wh_rate = None

    for r in range(1, row_len, 1):
        x_piece = x[:, :, r:r+1]

        if l is None and torch.sum(x_piece) >= min_brightness:
            l = r
        elif l is not None and (torch.sum(x_piece) < min_brightness or r==row_len-1):
            wh_rate = (r-l) / col_len

            if recombination and sep_idxs != [] and \
                wh_rate < wh_rate_min and before_wh_rate < wh_rate_min and \
                (r-sep_idxs[-1][2]) / col_len < wh_rate_max and l-before_r < min_space:

                sep_idxs_temp.extend([sep_idxs[-1], [0, col_len, l, r]])
                l = sep_idxs[-1][2]
                del sep_idxs[-1]

            sep_idxs.append([0, col_len, l, r])

            l = None
            before_r = r
            before_wh_rate = wh_rate

    return sep_idxs


def plot_cutting_info(x, letter_idxs, block=True):
    fig = plt.figure()
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
    n_col = 3
    n_row = len(letter_idxs)//n_col+1
    cmap = 'gray' # plt.cm.gray_r

    axes = fig.add_subplot(n_col, 1, 1, xticks=[], yticks=[])
    axes.imshow(x[0], cmap=cmap)
    axes.text(0, -50, f'판단한 글자 수: {len(letter_idxs)}', size=50)
    
    for idx, (yt, yb, xl, xr) in enumerate(letter_idxs):
        x_piece = x[:, yt:yb, xl:xr]
        wh_rate = (xr-xl) / (yb-yt)
        axes = fig.add_subplot(n_col+1, n_row, n_row+idx+1, xticks=[], yticks=[])
        axes.text(0, -15, f'{wh_rate:.2f}', size=12, ha="center", va="top", color='red' if wh_rate < 0.33 else 'black')
        axes.imshow(x_piece[0], cmap=cmap)

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
    sep_idxs = crop_by_separating_letter(x)
    sep_idxs2 = crop_by_separating_letter(x, recombination=False)
    # croped_idxs2 = crop_blank(sep_idxs)

    # print(sep_idxs, croped_idxs2, t)
    plot_cutting_info(x, sep_idxs, block=False)
    plot_cutting_info(x, sep_idxs2)
    input()

# print(croped_idxs2)
