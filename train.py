from data import KoSyllableDataset
from model import KoCtoP
from utils.plot import set_font
from utils.rich import new_progress, console
from utils.unicode import join_jamos

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import math

from rich import print
from rich.traceback import install
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device\n".format(device))


batch_size = 10
train_set = KoSyllableDataset()
train_loader = DataLoader(train_set, batch_size, shuffle=True)


model = KoCtoP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(model, train_loader, loss_fn, optimizer, progress):
    batch_size = train_loader.batch_size
    iters = math.ceil(len(train_set) // batch_size)

    task_id = progress.add_task(f'iter {batch_size}/{iters}', total=iters)

    size = len(train_loader.dataset)
    train_loss, correct = 0, 0

    for iter, (x, t) in enumerate(train_loader):
        x = x.to(device)
        yi, ym, yf = model(x)
        yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
        
        ti, tm, tf = t.values()
        loss = loss_fn(yi, ti) + loss_fn(ym, tm) + loss_fn(yf, tf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        ones = torch.ones([batch_size])
        mask_i = (yi.argmax(1) == ti)
        mask_m = (ym.argmax(1) == tm)
        mask_f = (yf.argmax(1) == tf)
        correct += (ones * mask_i * mask_m * mask_f).sum().item()

        if iter % 10 == 0:
            loss, current = loss.item(), ((iter+1) * len(x))
            progress.log(f"loss: {loss:>6f}")

        progress.update(task_id, description=f'iter {current}/{size}', advance=1)
    
    progress.remove_task(task_id)
    train_loss /= size
    correct /= size
    return train_loss, correct


epochs = 10

with new_progress() as progress:
    task_id = progress.add_task(f'epoch 1/{epochs}', total=epochs)

    for epoch in range(1, epochs+1):
        train_loss, accuracy = train(model, train_loader, loss_fn, optimizer, progress)
        
        progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)
