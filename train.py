import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import time
import argparse
import pandas as pd
import numpy as np
from rich.traceback import install
install()

from data import KoSyllableDataset
from kohwctop.model import KoCtoPSmall, KoCtoP
from utils.plot import set_font
from utils.rich import new_progress, console
from utils.utils import read_csv
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
                        default=1e-3,
                        help='학습률 (Learning Rate)')
args = parser.parse_args()


set_font(family='BM JUA_TTF')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = "cuda" if torch.cuda.is_available() else "cpu"
console.log("Using [green]{}[/green] device\n".format(device))


class Trainer:
    MODEL_NAME = 'model.pt'
    TRAIN_STEP_RESULT_PATH = "train_step_result.csv"
    train_step_result = {'n_learn': [], 'loss': [], 'acc': []}

    def __init__(self, train_loader, loss_fn, optimizer, device, print_every, save_dir):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.print_every = print_every
        self.save_dir = save_dir
        self.progress = new_progress()

        if os.path.exists(save_dir+Trainer.TRAIN_STEP_RESULT_PATH):
            Trainer.train_step_result = read_csv(save_dir+Trainer.TRAIN_STEP_RESULT_PATH, return_dict=True)
            self.n_learn = Trainer.train_step_result['n_learn'][-1]
        else:
            self.n_learn = 0

    def train(self, model, epochs):
        self.progress.start()
        task_id = self.progress.add_task(f'epoch 1/{epochs}', total=epochs)
    
        for epoch in range(1, epochs+1):
            train_loss, train_acc = self.train_epoch(model)
            test_loss, test_acc = test(model, test_loader, self.loss_fn, self.progress)
            
            self.progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)
            self.save_model(model, test_acc, test_loss)
        
        self.progress.stop()

    def train_epoch(self, model):
        model.train()
        start_iter = self.n_learn+self.train_loader.batch_size
        size = len(self.train_loader.dataset)

        task_id = self.progress.add_task(f'iter {start_iter}/{size}', total=size)
    
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
            self.n_learn += len(x)
    
            self.progress.update(task_id, description=f'iter {(self.n_learn % size) + current}/{size}', advance=len(x))
            
            if (iter+1) % self.print_every == 0:
                avg_loss = train_loss_per_print / current_per_print
                avg_acc = correct_per_print / current_per_print * 100
                self.progress.log(f"loss: {avg_loss:>6f} | acc: {avg_acc:>0.1f}%")
                self.save_step_result(avg_loss, avg_acc)
                train_loss_per_print = 0
                correct_per_print = 0
                current_per_print = 0
                
            if current % 10000 == 0:
                avg_loss = train_loss / current
                avg_acc = correct / current * 100
                self.save_model(model, avg_acc, avg_loss)
                
        self.progress.remove_task(task_id)
        train_loss /= current
        correct /= current
        return train_loss, correct * 100
    
    def save_model(self, model, acc, loss):
        torch.save(model.state_dict(), self.save_dir+Trainer.MODEL_NAME)
        self.progress.log(f'Saved PyTorch Model State to {self.save_dir+Trainer.MODEL_NAME}')

    def save_step_result(self, loss: float, acc: float) -> None:
        Trainer.train_step_result["n_learn"].append(self.n_learn)
        Trainer.train_step_result["loss"].append(f'{loss:>6f}')
        Trainer.train_step_result["acc"].append(f'{acc:>0.1f}')
        
        file_name = Trainer.TRAIN_STEP_RESULT_PATH
        train_step_df = pd.DataFrame(Trainer.train_step_result)
        train_step_df.to_csv(self.save_dir+file_name, encoding="UTF-8", index=False)


train_set = KoSyllableDataset()
test_set = KoSyllableDataset(train=False)
train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, args.batch_size, shuffle=True)
console.log(f'데이터 로드 완료! (train set: {len(train_set)} / test set: {len(test_set)})')


file_path = args.load_model
save_datetime = file_path.split('/')[0] if file_path else \
    time.strftime('%Y%m%d_%H%M%S', time.localtime())

save_dir = f'save/{save_datetime}/'
os.makedirs(save_dir, exist_ok=True)

model = KoCtoP().to(device)
if file_path:
    model.load_state_dict(torch.load(save_dir+file_path.split('/')[1]))
console.log('모델 {} 완료!'.format('로드' if file_path else '준비'))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
console.log('lr:', np.format_float_scientific(args.lr, exp_digits=1))


trainer = Trainer(train_loader, loss_fn, optimizer, device, args.print_every, save_dir)
trainer.train(model, args.epochs)
