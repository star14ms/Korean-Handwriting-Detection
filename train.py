from data import KoSyllableDataset
from model import KoCtoP
from utils.plot import set_font
from utils.rich import new_progress, console

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import math

from rich.traceback import install
install()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = "cuda" if torch.cuda.is_available() else "cpu"
console.log("Using [green]{}[/green] device\n".format(device))


batch_size = 50
train_set = KoSyllableDataset() # syllable: 음절
test_set = KoSyllableDataset(train=False)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)
console.log(f'데이터 로드 완료! (train set: {len(train_set)} / test set: {len(test_set)})')


model = KoCtoP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
console.log(f'모델 준비 완료!')


def train(model, train_loader, loss_fn, optimizer, progress):
    model.train()
    size = len(train_loader.dataset)
    batch_size = train_loader.batch_size

    task_id = progress.add_task(f'iter {batch_size}/{size}', total=size)

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

        current = (iter+1) * len(x)
        progress.update(task_id, description=f'iter {current}/{size}', advance=1)
        
        if iter % 10 == 0:
            progress.log(f"loss: {loss.item():>6f}")

    
    progress.remove_task(task_id)
    train_loss /= size
    correct /= size * 100
    return train_loss, correct


def test(model, test_loader, loss_fn, progress):
    model.eval()
    batch_size = test_loader.batch_size
    size = len(test_loader.dataset)
    task_id = progress.add_task(f'iter {batch_size}/{size}', total=size)

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for iter, (x, t) in enumerate(test_loader):
            x = x.to(device)
            yi, ym, yf = model(x)
            yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
            
            ti, tm, tf = t.values()
            loss = loss_fn(yi, ti) + loss_fn(ym, tm) + loss_fn(yf, tf)
            test_loss += loss.item()
            
            ones = torch.ones([batch_size])
            mask_i = (yi.argmax(1) == ti)
            mask_m = (ym.argmax(1) == tm)
            mask_f = (yf.argmax(1) == tf)
            correct += (ones * mask_i * mask_m * mask_f).sum().item()
    
            current = (iter+1) * len(x)
            progress.update(task_id, description=f'iter {current}/{size}', advance=1)

            if iter % 10 == 0:
                progress.log(f"loss: {loss.item():>6f}")

    test_loss /= size
    correct /= size * 100
    progress.log(f"Test Error: \n Accuracy: {(correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    progress.remove_task(task_id)

    return test_loss, correct


epochs = 1

with new_progress() as progress:
    task_id = progress.add_task(f'epoch 1/{epochs}', total=epochs)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, progress)
        test_loss, test_acc = test(model, train_loader, loss_fn, progress)
        
        progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)

        file_name = model.__class__.__name__ + f' Acc_{(test_acc):>0.2f}-Avg loss_{test_loss:>8f}.pth'
        torch.save(model.state_dict(), file_name)
        
        progress.log(f'Saved PyTorch Model State to {file_name}.pth')
