import os

import torch
from torch.nn import CTCLoss
from torch.autograd import Variable

from baseline.data import loadData
from baseline.evaluation import evaluation_metrics
from baseline.save import save_model
from utils.rich import console


def train(num_epochs, model, device, train_loader, val_loader, images, texts, lengths, converter, optimizer, lr_scheduler, prediction_dir, progress, save_desc='', start_epoch=0) :
    criterion = CTCLoss()
    criterion.to(device)
    images = images.to(device)
    model.to(device)
    size = len(train_loader.dataset)

    task_id = progress.add_task(f'epoch {start_epoch+1}/{num_epochs}', total=num_epochs)

    for epoch in range(start_epoch+1, num_epochs+1):
        train_epoch(model, train_loader, images, texts, lengths, converter, criterion, optimizer, size, progress)

        # 검증
        validation(model, device, val_loader, images, texts, lengths, converter, prediction_dir, progress)
        
        save_name = '{}{}'.format(epoch, save_desc)
        save_model(save_name, model, optimizer, lr_scheduler)
        
        lr_scheduler.step()

        progress.update(task_id, description=f'epoch {epoch}/{num_epochs}', advance=1)
        progress.log(f'Saved PyTorch Model State to save/{save_name}')
    
    progress.remove_task(task_id)


def train_epoch(model, train_loader, images, texts, lengths, converter, criterion, optimizer, size, progress):
    current = 0
    model.train()

    subtask_id = progress.add_task(f'train {train_loader.batch_size}/{size}', total=size)

    for i, datas in enumerate(train_loader) :
        datas, targets = datas
        batch_size = datas.size(0)
        loadData(images, datas)
        t, l = converter.encode(targets)
        loadData(texts, t)
        loadData(lengths, l)
        
        # 모델 학습 진행
        preds = model(images)
        # print(targets)
        # print(t.shape, l)
        # print(datas.shape + '->' + preds.shape))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        # loss 계산->back-propagation
        cost = criterion(preds, texts, preds_size, lengths) / batch_size
        model.zero_grad()
        cost.backward()
        optimizer.step()
        
        current += len(datas)
        progress.update(subtask_id, description=f'train {current}/{size}', advance=len(images))

        if i % 10 == 0:
            progress.log(f"loss: {cost.item()/len(images):>6f}")

    progress.remove_task(subtask_id)


def validation(model, device, val_loader, images, texts, lengths, converter, prediction_dir, progress):
    image_path_list, pred_list = test(model, device, val_loader, images, texts, lengths, converter, prediction_dir, progress)
    console.log('validation test finish')

    # 추론 점수 계산
    res = evaluation_metrics(pred_list, val_loader.dataset)

    console.log('validation : {:>6f}'.format(res))
    
    
def test(model, device, test_loader, images, texts, lengths, converter, prediction_dir, progress) :
    model.to(device)
    images = images.to(device)
    model.eval()
    image_path_list = test_loader.dataset.imgs
    pred_list = []
    current = 0
    size = len(test_loader.dataset)

    task_id = progress.add_task(f'test {test_loader.batch_size}/{size}', total=size)
    
    os.makedirs(os.path.join(prediction_dir), exist_ok=True)
    for i, datas in enumerate(test_loader) :
        datas, targets = datas
        batch_size = datas.size(0)
        loadData(images, datas)
        t, l = converter.encode(targets)
        loadData(texts, t)
        loadData(lengths, l)

        # 추론
        with torch.no_grad():
            preds = model(images)
        
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # 추론 결과 decode
        pred_string = converter.decode(preds.data, preds_size.data, raw=False)
        
        # 추론 결과 list로 저장
        if type(pred_string) is str:
            pred_list.append(pred_string)
        else:
            pred_list.extend(pred_string)
        
        current += len(images)
        progress.update(task_id, description=f'test {current}/{size}', advance=len(images))

    progress.remove_task(task_id)
    
    return image_path_list, pred_list
