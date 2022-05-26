import matplotlib.pyplot as plt 
import torch

from rich import print
from rich.console import Console
from rich.traceback import install
from rich.progress import track

from data import HWKoDataset
from plot import set_font
install()
set_font(family='BM JUA_TTF')
console = Console()


def crop_by_separating_letter(x, min_brightness=3, wh_rate_range=(0.2, 2)):
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

    croped_idxs = []
    l_idx = None

    for r_idx in range(1, row_len, 1):
        x_piece = x[:, :, r_idx:r_idx+1]

        if l_idx is None and torch.sum(x_piece) >= min_brightness:
            l_idx = r_idx
        elif l_idx is not None and (torch.sum(x_piece) < min_brightness or r_idx==row_len-1):
            # wh_rate = (r_idx-l_idx) / col_len

            # if wh_rate < wh_rate_range[0] and croped_idxs != [] and \
            #     (r_idx-croped_idxs[-1][2]) / col_len < wh_rate_range[1]:
            #     l_idx = croped_idxs[-1][2]
            #     del croped_idxs[-1]
            #     croped_idxs.append([0, col_len, l_idx, r_idx])
            # else:
            #     croped_idxs.append([0, col_len, l_idx, r_idx])
            croped_idxs.append([0, col_len, l_idx, r_idx])
            l_idx = None
            
    return croped_idxs


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
        axes = fig.add_subplot(n_col+1, n_row, n_row+idx+1, xticks=[], yticks=[])
        axes.imshow(x_piece[0], cmap=cmap)

    plt.show(block=block)


def crop_blank(x, base_letter_idxs, min_brightness=3):
    base_yt, _, base_xl, _ = base_letter_idxs
    row_len = x.shape[2] # (C, H, (W))
    col_len = x.shape[1] # (C, (H), W)

    yt = 0
    while torch.sum(x[:, yt:yt+1, :]) < min_brightness and yt < col_len:
        yt += 1

    yb = col_len
    while torch.sum(x[:, yb:yb+1, :]) < min_brightness and yb >= 0:
        yb -= 1

    xl = 0
    while torch.sum(x[:, :, xl:xl+1]) < min_brightness and xl < row_len:
        xl += 1

    xr = row_len
    while torch.sum(x[:, :, xr:xr+1]) < min_brightness and xr >= 0:
        xr -= 1

    return (base_yt + yt, base_yt + yb, base_xl + xl, base_xl + xr)


train_set = HWKoDataset()


for x, t in track(train_set, total=len(train_set)):
    croped_idxs = crop_by_separating_letter(x)
        
    plot_cutting_info(x, croped_idxs, block=False)

    croped_idxs2 = []
    for idx, base_letter_idxs in enumerate(croped_idxs):
        yt, yb, xl, xr = base_letter_idxs
        x_piece = x[:, yt:yb, xl:xr]
        new_crop_idx = crop_blank(x_piece, base_letter_idxs)
        croped_idxs2.append(new_crop_idx)

    print(croped_idxs, croped_idxs2, t)
    plot_cutting_info(x, croped_idxs2)

print(croped_idxs2)
