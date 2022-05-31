from data import KoSyllableDataset
from model import KoCtoP

import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import argparse

from utils.rich import new_progress, console
from utils.plot import show_img_and_scores, set_font
from tools.constant import label_to_syllable


parser = argparse.ArgumentParser()
parser.add_argument('--load-model', type=str, dest='load_model',
                        default='save/KoCtoP-acc_14.285%-loss_0.088537-420000.pth',
                        help='model weight path to load')
parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=50,
                        help='batch size')
args = parser.parse_args()


def test(model, test_loader, loss_fn, progress, show_wrong_info=False):
    model.eval()
    batch_size = test_loader.batch_size
    size = len(test_loader.dataset)
    task_id = progress.add_task(f'iter {batch_size}/{size}', total=size)

    test_loss, correct, current = 0, 0, 0
    
    with torch.no_grad():
        for iter, (x, t) in enumerate(test_loader):
            x = x.to(device)
            yi, ym, yf = model(x)
            yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
            pi, pm, pf = yi.argmax(1), ym.argmax(1), yf.argmax(1)
            
            ti, tm, tf = t.values()
            loss = loss_fn(yi, ti) + loss_fn(ym, tm) + loss_fn(yf, tf)
            test_loss += loss.item()
            
            ones = torch.ones([len(x)])
            mask_i = (pi == ti)
            mask_m = (pm == tm)
            mask_f = (pf == tf)
            correct_info = ones * mask_i * mask_m * mask_f
            n_correct = correct_info.sum().item()
            correct += n_correct

            if show_wrong_info and n_correct != len(x):
                for idx in torch.where(correct_info == False)[0]:
                    char_y = label_to_syllable(
                        pi[idx].item(), 
                        pm[idx].item(), 
                        pf[idx].item()
                    )
                    char_t = label_to_syllable(
                        ti[idx].item(), 
                        tm[idx].item(), 
                        tf[idx].item()
                    )
                    text = '예측: {} 정답: {}'.format(char_y, char_t)
                    show_img_and_scores(x[idx].cpu(), yi[idx], ym[idx], yf[idx], title=text)

            current += len(x)
            progress.update(task_id, description=f'iter {current}/{size}', advance=len(x))

            if iter % 10 == 0:
                progress.log(f"loss: {test_loss/current:>6f} acc: {correct/current*100:>0.3f}%")

    test_loss /= current
    correct /= current
    progress.log(f"Test Error: \n Accuracy: {correct*100:>0.3f}%, Avg loss: {test_loss:>6f} \n")
    progress.remove_task(task_id)

    return test_loss, correct * 100


def predict(x, t, model, device, plot=True, description=None):
    model.eval()
    x = x.unsqueeze(0).to(device)
    
    yi, ym, yf = model(x)
    yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
    ti, tm, tf = t.values()
    pi, pm, pf = yi.argmax(1), ym.argmax(1), yf.argmax(1)

    char_y = label_to_syllable(pi.item(), pm.item(), pf.item())
    
    if t is not None:
        char_t = label_to_syllable(ti, tm, tf)
        correct = (pi==ti and pm==tm and pf==tf)
    
        text = 'test data {} - 예측: {} 정답: {}'.format(description, char_y, char_t)
    else:
        text = 'test data {} - 예측: {}'.format(description, char_y)
    
    if plot:
        show_img_and_scores(x.cpu(), yi, ym, yf, title=text)
    else:
        console.log(text)
        input()
    
    if t is not None:
        return char_y, correct
    else:
        return char_y


def test_sample(test_set, model, device, random_sample=True):
    model.eval()
    with torch.no_grad():
        idx = -1
        while True:
            if random_sample:
                idx = random.randint(0, len(test_set)-1) 
            else:
                try: 
                    idx = int(input('data idx: '))
                except ValueError:
                    idx += 1
                    
            predict(*test_set[idx], model, device, description=idx)


if __name__ == '__main__':
    set_font(family='BM JUA_TTF')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.log("Using [green]{}[/green] device\n".format(device))
    
    batch_size = args.batch_size
    test_set = KoSyllableDataset(train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    model = KoCtoP().to(device)
    model.load_state_dict(torch.load(args.load_model))
    
    console.log('모델 로드 완료!')

    model.show_feature_maps(test_set, 0, device)

    test_sample(test_set, model, device, random_sample=False)
    
    # loss_fn = nn.CrossEntropyLoss()
    # with new_progress() as progress:
        # test(model, test_loader, loss_fn, progress, show_wrong_info=False)
