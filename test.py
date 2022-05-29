from utils.unicode import join_jamos
from utils.rich import console


def predict(dataset, idx, model, device, verbose=True):
    model.eval()
    x, t = dataset[idx]
    x = x.unsqueeze(0).to(device)

    yi, ym, yf = model(x)
    pi, pm, pf = yi.argmax(1).cpu(), \
            ym.argmax(1).cpu(), \
            yf.argmax(1).cpu()

    chr_yi = dataset.to_chr['i'][pi.item()]
    chr_ym = dataset.to_chr['m'][pm.item()]
    chr_yf = dataset.to_chr['f'][pf.item()]
    char = join_jamos(chr_yi + chr_ym + chr_yf)

    if t is not None:
        label_yi = dataset.to_chr['i'][t['initial']]
        label_ym = dataset.to_chr['m'][t['medial']]
        label_yf = dataset.to_chr['f'][t['final']]
    
        if verbose:
            console.print('예측: {} 정답: {}'.format(char, join_jamos(label_yi + label_ym + label_yf)))
    
    elif verbose:
        console.print('예측: {}'.format(char))

    return char
