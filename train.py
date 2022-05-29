from data import KoSyllableDataset
from utils.plot import set_font
from utils.rich import new_progress, console

import torch
from torch.utils.data import DataLoader
import random
import os

from rich import print
from rich.traceback import install
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


train_set = KoSyllableDataset()
train_loader = DataLoader(train_set, batch_size=100, shuffle=False)


for x, t in train_loader:
    print(t)
    input()