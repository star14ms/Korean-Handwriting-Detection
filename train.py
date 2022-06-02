from data import KoSyllableDataset
from model import KoCtoP
from utils.plot import set_font
from utils.rich import new_progress, console
from utils.utils import makedirs
from test import test

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse

from rich.traceback import install
install()


parser = argparse.ArgumentParser()
parser.add_argument('--load-model', type=str, dest='load_model',
                        default=None,
                        help='이어서 학습시킬 모델 경로 (model weight path to load)')
parser.add_argument('--epoch', type=int, dest='epochs',
                        default=1,
                        help='학습 시킬 바퀴 수 (num epochs)')
parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=50,
                        help='묶어서 학습할 숫자 (batch size)')
args = parser.parse_args()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = "cuda" if torch.cuda.is_available() else "cpu"
console.log("Using [green]{}[/green] device\n".format(device))


batch_size = args.batch_size
train_set = KoSyllableDataset()
test_set = KoSyllableDataset(train=False)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)
console.log(f'데이터 로드 완료! (train set: {len(train_set)} / test set: {len(test_set)})')


save_dir = 'save/'
makedirs(save_dir)

file_name = args.load_model
start = int(file_name.split('-')[-1].replace('.pth','')) if file_name else 0
model = KoCtoP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
if file_name:
    model.load_state_dict(torch.load(save_dir+file_name))
console.log('모델 {} 완료!'.format('로드' if file_name else '준비'))


def train(model, train_loader, loss_fn, optimizer, progress):
    model.train()
    size = len(train_loader.dataset)
    batch_size = train_loader.batch_size

    task_id = progress.add_task(f'iter {batch_size}/{size}', total=size)

    train_loss, correct, current = 0, 0, 0

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
        
        ones = torch.ones([len(x)])
        mask_i = (yi.argmax(1) == ti)
        mask_m = (ym.argmax(1) == tm)
        mask_f = (yf.argmax(1) == tf)
        correct += (ones * mask_i * mask_m * mask_f).sum().item()

        current += len(x)
        progress.update(task_id, description=f'iter {current}/{size}', advance=len(x))
        
        if iter % 10 == 0:
            progress.log(f"loss: {loss.item()/batch_size:>6f}")

        if current % 10000 == 0:
            avg_loss = train_loss / current
            avg_acc = correct / current * 100
            file_name = model.__class__.__name__ + f'-acc_{(avg_acc):>0.3f}%-loss_{avg_loss:>6f}-{start+current}.pth'
            torch.save(model.state_dict(), save_dir+file_name)
        
            progress.log(f'Saved PyTorch Model State to {save_dir+file_name}')
    
    progress.remove_task(task_id)
    train_loss /= current
    correct /= current
    return train_loss, correct * 100


epochs = args.epochs

with new_progress() as progress:
    task_id = progress.add_task(f'epoch 1/{epochs}', total=epochs)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, progress)
        test_loss, test_acc = test(model, train_loader, loss_fn, progress)
        
        progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)

        file_name = model.__class__.__name__ + f'-acc_{(test_acc):>0.3f}%-loss_{test_loss:>6f}.pth'
        torch.save(model.state_dict(), save_dir+file_name)
        
        progress.log(f'Saved PyTorch Model State to {save_dir+file_name}')
