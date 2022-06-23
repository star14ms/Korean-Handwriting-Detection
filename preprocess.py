from data import KoHWSentenceDataset
from utils.plot import set_font
from kohwctop.preprocess.functions import (
    get_data_from_train_set,
    save_n_piece,
    get_corrct_rate_n_piece,
    separate_by_space,
    merge_pieces,
    crop_blank,
)
from insight.data_insight import (
    plot_sentence_cutting_info,
    plot_brightness_gradient,
)

import torch
import random
import os

from rich import print
from rich.progress import track
from rich.traceback import install
from utils.rich import new_progress, console
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


train_set = KoHWSentenceDataset()
n_corrct_n_seperated = 0

kwargs = {
    'kernel_width': 10,
    'min_brightness': 20,
    'min_space': 1,
    'min_letter_len': 1,
}


for x, t in track(train_set, total=len(train_set)):
    # x, t = random.choice(train_set)
    # x, t = train_set[0]
    sep_idxs, brightness_list = separate_by_space(x, **kwargs)
    # sep_idxs = merge_pieces(sep_idxs)
    # sep_idxs2 = crop_blank(x, sep_idxs)

    plot_sentence_cutting_info(x, sep_idxs, t, block=False)

    title = f'Image Brightness Gradient\n(kernel_width: {kwargs["kernel_width"]})'
    plot_brightness_gradient(x, brightness_list, title=title, ylim=100, block=True)
    # input()


# with new_progress() as progress:
#     n_pieces_set = get_data_from_train_set(train_set, progress, func=save_n_piece, **kwargs)

# correct_rate = get_corrct_rate_n_piece(train_set, n_pieces_set)
# print(f'{correct_rate:.2f}% correct')
