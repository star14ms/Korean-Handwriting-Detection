import torch
import os
import pandas as pd

from utils.rich import new_progress
from utils.utils import read_csv
from test import test


class Trainer:
    MODEL_NAME = 'model.pt'
    TRAIN_STEP_RESULT_PATH = "train_step_result.csv"
    train_step_result = {'n_learn': [], 'loss': [], 'acc': [], 'morp_acc': []}

    def __init__(self, train_loader, test_loader, loss_fn, optimizer, device, print_every, save_dir):
        self.train_loader = train_loader
        self.test_loader = test_loader
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
            test_loss, test_acc = test(model, self.test_loader, self.loss_fn, self.progress)
            
            self.progress.update(task_id, description=f'epoch {epoch}/{epochs}', advance=1)
            self.save_model(model, test_loss, test_acc, -1)
        
        self.progress.stop()

    def train_epoch(self, model):
        model.train()
        start_iter = self.n_learn+self.train_loader.batch_size
        size = len(self.train_loader.dataset)

        task_id = self.progress.add_task(f'iter {start_iter}/{size}', total=size)
        
        train_loss, train_loss_local = 0, 0
        correct, correct_local = 0, 0
        m_correct, m_correct_local = 0, 0
        current, current_local = 0, 0
        m_current, m_current_local = 0, 0
    
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
            train_loss_local += loss.item()
            
            ones = torch.ones([len(x)])
            mask_i = (yi.argmax(1) == ti)
            mask_m = (ym.argmax(1) == tm)
            mask_f = (yf.argmax(1) == tf)
            correct_batch = (ones * mask_i * mask_m * mask_f).sum().item()
            m_corrent_batch = mask_i.sum().item() + mask_m.sum().item() + mask_f.sum().item()
            
            correct += correct_batch
            correct_local += correct_batch
            m_correct += m_corrent_batch
            m_correct_local += m_corrent_batch
            current += len(x)
            current_local += len(x)
            m_current += len(x) * 3
            m_current_local += len(x) * 3
            self.n_learn += len(x)
    
            self.progress.update(task_id, description=f'iter {self.n_learn % size}/{size}', advance=len(x))
            
            if (iter+1) % self.print_every == 0:
                avg_loss = train_loss_local / current_local
                avg_acc = correct_local / current_local * 100
                avg_m_acc = m_correct_local / m_current_local * 100
                self.progress.log(f"loss: {avg_loss:>6f} | acc: {avg_acc:>0.1f}% | morp acc: {avg_m_acc:>0.1f}%")
                self.save_step_result(avg_loss, avg_acc, avg_m_acc)
                train_loss_local = 0
                correct_local = 0
                m_correct_local = 0
                current_local = 0
                m_current_local = 0
                
            if current % 10000 == 0:
                avg_loss = train_loss / current
                avg_acc = correct / current * 100
                avg_m_acc = m_correct / m_current * 100
                self.save_model(model, avg_loss, avg_acc, avg_m_acc)
                
        self.progress.remove_task(task_id)
        train_loss /= current
        correct /= current
        return train_loss, correct * 100
    
    def save_model(self, model, loss, acc, m_acc):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(model.state_dict(), self.save_dir+Trainer.MODEL_NAME)
        self.progress.log(f'Saved PyTorch Model State to {self.save_dir+Trainer.MODEL_NAME}')

    def save_step_result(self, loss: float, acc: float, m_acc: float) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        Trainer.train_step_result["n_learn"].append(self.n_learn)
        Trainer.train_step_result["loss"].append(f'{loss:>6f}')
        Trainer.train_step_result["acc"].append(f'{acc:>0.1f}')
        Trainer.train_step_result["morp_acc"].append(f'{m_acc:>0.1f}')
        
        file_name = Trainer.TRAIN_STEP_RESULT_PATH
        train_step_df = pd.DataFrame(Trainer.train_step_result)
        train_step_df.to_csv(self.save_dir+file_name, encoding="UTF-8", index=False)
