from data import HWKoDataset
from utils.plot import set_font
from preprocessing.functions import (
    get_data_from_train_set,
    save_n_piece,
    get_corrct_rate_n_piece,
    separate_by_space,
    crop_blank,
)
from data_insight import (
    plot_cutting_data_info,
    plot_brightness_gradient,
)

import torch
import random
import os

from rich import print
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from utils.rich import new_progress
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

console = Console()
 

train_set = HWKoDataset()
n_corrct_n_seperated = 0

kwargs = {
    'kernel_width': 10,
    'min_brightness': 20,
    'min_space': 1,
    'min_letter_len': 30,
}

# with new_progress() as progress:
#     n_pieces_set = get_data_from_train_set(train_set, progress, func=save_n_piece, **kwargs)

# correct_rate = get_corrct_rate_n_piece(train_set, n_pieces_set)
# print(f'{correct_rate:.2f}% correct')


for x, t in track(train_set, total=len(train_set)):
    # x, t = random.choice(train_set)
    # x, t = train_set[0]
    sep_idxs, brightness_list = separate_by_space(x, **kwargs)

    # sep_idxs2 = crop_blank(x, sep_idxs)

    # plot_cutting_data_info(x, sep_idxs, t, block=True)
    title = f'Brightness Gradient\n(kernel_width: {kwargs["kernel_width"]})'
    plot_brightness_gradient(x, brightness_list, title=title, ylim=100, block=True)
    # input()
