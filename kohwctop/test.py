import torch
import random
import sys

from utils.rich import console
from utils.plot import show_img_and_scores, wide_plot_kwargs, imf_plot_kwargs
from tools.constant import label_to_syllable, to_char


@torch.no_grad()
def test(model, test_loader, loss_fn, progress, print_every, show_wrong_info=False):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    batch_size = test_loader.batch_size
    size = len(test_loader.dataset)
    task_id = progress.add_task(f'iter {batch_size}/{size}', total=size)

    test_loss, correct, current = 0, 0, 0
    i_correct, m_correct, f_correct = 0, 0, 0

    def make_log(avg_loss, avg_acc, avg_i_acc, avg_m_acc, avg_f_acc):
        return 'loss: {:>6f} | acc: {:>0.2f}% (초성 {:>0.1f}%) (중성 {:>0.1f}%) (종성 {:>0.1f}%)'.format(
            avg_loss, avg_acc, avg_i_acc, avg_m_acc, avg_f_acc
        )

    for iter, (x, t) in enumerate(test_loader):
        x = x.to(device)
        yi, ym, yf = torch._to_cpu(model(x))
        pi, pm, pf = yi.argmax(1), ym.argmax(1), yf.argmax(1)
        
        ti, tm, tf = t.values()
        loss = loss_fn(yi, ti) + loss_fn(ym, tm) + loss_fn(yf, tf)
        
        ones = torch.ones([len(x)])
        mask_i = (pi == ti)
        mask_m = (pm == tm)
        mask_f = (pf == tf)
        correct_info = ones * mask_i * mask_m * mask_f
        correct_batch = correct_info.sum().item()
        
        test_loss += loss.item()
        correct += correct_batch
        i_correct += mask_i.sum().item()
        m_correct += mask_m.sum().item()
        f_correct += mask_f.sum().item()
        current += len(x)

        if show_wrong_info and correct_batch != len(x):
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

        progress.update(task_id, description=f'iter {current}/{size}', advance=len(x))

        if (iter+1) % print_every == 0:
            avg_loss = test_loss / current
            avg_acc = correct / current * 100
            avg_i_acc = i_correct / current * 100
            avg_m_acc = m_correct / current * 100
            avg_f_acc = f_correct / current * 100

            progress.log(make_log(avg_loss, avg_acc, avg_i_acc, avg_m_acc, avg_f_acc))

    test_loss /= current
    correct /= current
    progress.log(f"Test Error: \n Accuracy: {correct*100:>0.3f}%, Avg loss: {test_loss:>6f} \n")
    progress.remove_task(task_id)

    return test_loss, correct * 100


def get_top5_hangul_char(yi, ym, yf):
    top5 = {}
    yi_top5 = yi.topk(5)
    ym_top5 = ym.topk(5)
    yf_top5 = yf.topk(5)
    yi_top5_scores, yi_top5_idxs = yi_top5[0], yi_top5[1]
    ym_top5_scores, ym_top5_idxs = ym_top5[0], ym_top5[1]
    yf_top5_scores, yf_top5_idxs = yf_top5[0], yf_top5[1]
    yi_idx, ym_idx, yf_idx = 0, 0, 0

    for top_n in range(1, 5+1):
        char_y = label_to_syllable(yi_top5_idxs[yi_idx].item(), ym_top5_idxs[ym_idx].item(), yf_top5_idxs[yf_idx].item()).strip()
        mean_score = torch.mean(torch.stack([yi_top5_scores[yi_idx], ym_top5_scores[ym_idx], yf_top5_scores[yf_idx]]))
        top5[char_y] = round(mean_score.item(), 1)
        
        if top_n == 5:
            break

        yi_delta_next = yi_top5_scores[yi_idx] - yi_top5_scores[yi_idx+1]
        ym_delta_next = ym_top5_scores[ym_idx] - ym_top5_scores[ym_idx+1]
        yf_delta_next = yf_top5_scores[yf_idx] - yf_top5_scores[yf_idx+1]

        if yi_delta_next < ym_delta_next and yi_delta_next < yf_delta_next:
            yi_idx += 1

        elif ym_delta_next < yi_delta_next and ym_delta_next < yf_delta_next:
            ym_idx += 1

        elif yf_delta_next < yi_delta_next and yf_delta_next < ym_delta_next:
            yf_idx += 1
        
    return top5

@torch.no_grad()
def predict_KoCtoP(x, t, model, plot=False, plot_when_wrong=True, description=None, verbose=False):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    while len(x.shape) < 4:
        x = x.unsqueeze(0)
    
    yi, ym, yf = model(x.to(device))
    yi, ym, yf = yi.cpu(), ym.cpu(), yf.cpu()
    pi, pm, pf = yi.argmax(1), ym.argmax(1), yf.argmax(1)

    yi = torch.softmax(yi[0], dim=0) * 100
    ym = torch.softmax(ym[0], dim=0) * 100
    yf = torch.softmax(yf[0], dim=0) * 100

    top5 = get_top5_hangul_char(yi, ym, yf)
    top1_char = tuple(top5.keys())[0]
    
    if t is not None:
        ti, tm, tf = t.values()
        char_t = label_to_syllable(ti, tm, tf)
        correct = (pi==ti and pm==tm and pf==tf)
    
        text = 'test data {} - 예측: {} 정답: {}'.format(description, top1_char, char_t)
    else:
        text = 'test data {} - 예측: {}'.format(description, top1_char)
        correct = None
    
    if plot and (not plot_when_wrong or (t is not None and not correct)):
        show_img_and_scores(x.cpu(), yi, ym, yf, ys_kwargs=imf_plot_kwargs, title=text)
    
    if verbose:
        color = 'green' if correct else 'white' if correct is None else 'red'
        text = f'[{color}]' + text + f'[/{color}]'
        console.log(text)
    
    if t is not None:
        return top5, correct
    else:
        return top5

@torch.no_grad()
def predict(x, t, model, model_KoCtoP, plot=False, plot_when_wrong=True, description=None, verbose=False):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    
    while len(x.shape) < 4:
        x = x.unsqueeze(0)
    
    y = model(x.to(device)).cpu()
    y = torch.softmax(y[0], dim=0) * 100
    p = y.argmax(0)

    top5 = {}
    top5_score_idx = y.topk(5)
    for idx, score in zip(top5_score_idx[1], top5_score_idx[0]):
        top5[to_char[idx.item()]] = round(score.item(), 1)
    
    top1_char = tuple(top5.keys())[0]
    
    if top5.get('한글 음절') is not None:
        del top5['한글 음절']

    if t is not None:
        ti, tm, tf = t.values()
        char_t = label_to_syllable(ti, tm, tf)
        correct = p == t
    
        text = 'test data {} - 예측: {} 정답: {}'.format(description, top1_char, char_t)
    else:
        text = 'test data {} - 예측: {}'.format(description, top1_char)
        correct = None
    
    if plot and (not plot_when_wrong or (t is not None and not correct)):
        show_img_and_scores(x.cpu(), y[:52], y[52:104], y[104:], ys_kwargs=wide_plot_kwargs, title=text)
    
    if top1_char == '한글 음절':
        return {**predict_KoCtoP(x, t, model_KoCtoP, plot, plot_when_wrong, description, verbose), **top5}
    
    if verbose:
        color = 'green' if correct else 'white' if correct is None else 'red'
        text = f'[{color}]' + text + f'[/{color}]'
        console.log(text)
    
    if t is not None:
        return top5, correct
    else:
        return top5

@torch.no_grad()
def test_sample(test_set, model, device, random_sample=True, plot_when_wrong=True, plot_feature_map=False):
    model.eval()
    idx = -1
    while True:
        if random_sample:
            idx = random.randint(0, len(test_set)-1) 
        else:
            try: 
                idx = int(input('data idx: '))
            except ValueError:
                idx += 1
                
        _, correct = predict(*test_set[idx], model, plot=True, plot_when_wrong=plot_when_wrong, description=idx)
        if (not correct or not plot_when_wrong) and plot_feature_map:
            model.show_feature_maps(test_set[idx][0], device, description=idx)

@torch.no_grad()
def predict_sentence(sentence_set, model):
    to_pil = sentence_set.to_pil
    to_tensor = sentence_set.to_tensor
    
    for x, _ in sentence_set:
        w = x.shape[2] # (C, H, W)
        h = x.shape[1]
        to_pil(x).show()
        before_pred = None
        same_pred_stack = 0

        for start_x in range(0, w-h+1, 2):
            x_piece = x[:,:,start_x:start_x+h]
            x_piece = to_pil(x_piece).resize((64, 64))
            x_piece = to_tensor(x_piece)
            pred = predict(x_piece, t=None, model=model)

            if pred == before_pred:
                same_pred_stack += 1
                if same_pred_stack == 2:
                    print(pred, end='')
            else:
                same_pred_stack = 0

            sys.stdout.flush()
            before_pred = pred
        
    input()
