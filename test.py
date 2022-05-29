from data import KoSyllableDataset
from model import KoCtoP

import torch
from torch.utils.data import DataLoader
import random

from utils.unicode import join_jamos
from utils.rich import console
from tools import to_chr


# file_name = ''


def predict(dataset, idx, model, device, verbose=True):
    model.eval()
    x, t = dataset[idx]
    x = x.unsqueeze(0).to(device)

    yi, ym, yf = model(x)
    pi, pm, pf = yi.argmax(1).cpu(), \
            ym.argmax(1).cpu(), \
            yf.argmax(1).cpu()

    chr_yi = to_chr['i'][pi.item()]
    chr_ym = to_chr['m'][pm.item()]
    chr_yf = to_chr['f'][pf.item()]
    char = join_jamos(chr_yi + chr_ym + chr_yf)

    if t is not None:
        label_yi = to_chr['i'][t['initial']]
        label_ym = to_chr['m'][t['medial']]
        label_yf = to_chr['f'][t['final']]
    
        if verbose:
            console.print('예측: {} 정답: {}'.format(char, join_jamos(label_yi + label_ym + label_yf)))
    
    elif verbose:
        console.print('예측: {}'.format(char))

    return char


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device\n".format(device))


batch_size = 10
test_set = KoSyllableDataset(train=False)
test_loader = DataLoader(test_set, batch_size, shuffle=True)


model = KoCtoP().to(device)
# model.load_state_dict(torch.load(file_name))


model.eval()

with torch.no_grad():
    while True:
        idx = random.randint(0, len(test_set)-1)
        predict(test_set, 0, model, device, verbose=True)