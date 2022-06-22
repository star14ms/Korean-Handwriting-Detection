import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
import pandas as pd
import numpy as np
from rich.traceback import install
install()

from data import KoSyllableDataset
from model import KoCtoP, KoCtoPLarge
from utils.plot import set_font
from utils.rich import new_progress, console
from utils.utils import makedirs
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('--load-model', type=str, dest='load_model',
                        default=None,
                        help='이어서 학습시킬 모델 경로 (model weight path to load)')
parser.add_argument('--epoch', type=int, dest='epochs',
                        default=1,
                        help='몇 바퀴 학습시킬 건지 (num epochs)')
parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=50,
                        help='묶어서 학습할 숫자 (batch size)')
parser.add_argument('--print-every', type=int, dest='print_every',
                        default=20,
                        help='학습 로그 출력하는 간격 (단위: batch))')
parser.add_argument('--lr', type=float, dest='lr',
                        default=1e-4,
                        help='학습률 (Learning Rate)')
args = parser.parse_args()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = "cuda" if torch.cuda.is_available() else "cpu"
console.log("Using [green]{}[/green] device\n".format(device))


train_set = KoSyllableDataset()
test_set = KoSyllableDataset(train=False)
train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, args.batch_size, shuffle=True)
console.log(f'데이터 로드 완료! (train set: {len(train_set)} / test set: {len(test_set)})')


save_dir = 'save/'
makedirs(save_dir)

file_name = args.load_model
start = int(file_name.split('-')[-1].replace('.pth','')) if file_name else 0
model = KoCtoPLarge().to(device)
if file_name:
    model.load_state_dict(torch.load(save_dir+file_name))
console.log('모델 {} 완료!'.format('로드' if file_name else '준비'))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
console.log('lr:', np.format_float_scientific(args.lr, exp_digits=1))


class Trainer:
    TRAIN_STEP_RESULT_PATH = "train_step_result.csv"
    train_step_result = {'n_learn':[], 'loss': [], 'acc': []}

    def __init__(self, train_loader, loss_fn, optimizer, device, print_every, save_dir):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.print_every = print_every
        self.save_dir = save_dir
        self.progress = new_progress()
  

    def train(self, model, batch_size, epochs):
        self.progress.start()
        task_id = self.progress.add_task(f'epoch 1/{epochs}', total=epochs)
    
        for epoch in range(1, epochs+1):
            train_loss, train_acc = self.train_epoch(model, batch_size)
            test_loss, test_acc = test(model, test_loader, self.loss_fn, self.progress)
            
            self.progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)
    
            file_name = model.__class__.__name__ + f'-acc_{(test_acc):>0.3f}%-loss_{test_loss:>6f}.pth'
            torch.save(model.state_dict(), self.save_dir+file_name)
            
            self.progress.log(f'Saved PyTorch Model State to {self.save_dir+file_name}')
        
        self.progress.stop()


    def train_epoch(self, model, batch_size):
        model.train()
        size = len(self.train_loader.dataset)
    
        task_id = self.progress.add_task(f'iter {batch_size}/{size}', total=size)
    
        train_loss, correct, current = 0, 0, 0
        train_loss_per_print, correct_per_print, current_per_print = 0, 0, 0
    
        for iter, (x, t) in enumerate(self.train_loader):
            x = x.to(self.device)
            yi, ym, yf = model(x)
            yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
            
            ti, tm, tf = t.values()
            loss = self.loss_fn(yi, ti) + self.loss_fn(ym, tm) + self.loss_fn(yf, tf)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item()
            train_loss_per_print += loss.item()
            
            ones = torch.ones([len(x)])
            mask_i = (yi.argmax(1) == ti)
            mask_m = (ym.argmax(1) == tm)
            mask_f = (yf.argmax(1) == tf)
            correct_batch = (ones * mask_i * mask_m * mask_f).sum().item()
            
            correct += correct_batch
            correct_per_print += correct_batch
            current += len(x)
            current_per_print += len(x)
    
            self.progress.update(task_id, description=f'iter {current}/{size}', advance=len(x))
            
            if (iter+1) % self.print_every == 0:
                avg_loss = train_loss_per_print / current_per_print
                avg_acc = correct_per_print / current_per_print * 100
                self.progress.log(f"loss: {avg_loss:>6f} | acc: {avg_acc:>0.1f}%")
                self.save_step_result(self.train_step_result, current, avg_loss, avg_acc)
                train_loss_per_print = 0
                correct_per_print = 0
                current_per_print = 0
                
            if current % 10000 == 0:
                avg_loss = train_loss / current
                avg_acc = correct / current * 100
                file_name = model.__class__.__name__ + f'-acc_{avg_acc:>0.2f}%-loss_{avg_loss:>6f}-{start+current}.pth'
                torch.save(model.state_dict(), self.save_dir+file_name)
            
                self.progress.log(f'Saved PyTorch Model State to {self.save_dir+file_name}')
        
        self.progress.remove_task(task_id)
        train_loss /= current
        correct /= current
        return train_loss, correct * 100
    

    def save_step_result(self, train_step_result: dict, current: int, loss: float, acc: float) -> None:
        train_step_result["n_learn"].append(current)
        train_step_result["loss"].append(f'{loss:>6f}')
        train_step_result["acc"].append(f'{acc:>0.1f}')
        
        train_step_df = pd.DataFrame(train_step_result)
        train_step_df.to_csv(self.save_dir+Trainer.TRAIN_STEP_RESULT_PATH, encoding="UTF-8", index=False)


trainer = Trainer(train_loader, loss_fn, optimizer, device, args.print_every, save_dir)
trainer.train(model, args.batch_size, args.epochs)
