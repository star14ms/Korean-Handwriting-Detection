import matplotlib.pyplot as plt
from matplotlib._color_data import TABLEAU_COLORS
import numpy as np

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from utils.utils import read_csv


TABLEAU_COLORS = TABLEAU_COLORS.values()


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def loss_graphs(losses_dic, smooth=True, ylim=None, xlabel='iterations', colors=TABLEAU_COLORS):
    keys = list(losses_dic.keys())
    for key, color in zip(keys, colors):
        x = np.arange(len(losses_dic[key]))
        y = smooth_curve(losses_dic[key]) if smooth else losses_dic[key]
        plt.plot(x, y, label=key) # f"-{color}"
    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.ylim(0, ylim)
    plt.legend(loc='upper right')
    plt.show()


def accuracy_graphs(accs_dic, ylim_min=None, xlabel='epochs', colors=TABLEAU_COLORS):
    markers = {'train': 'o', 'test': 's'}
    keys = list(accs_dic.keys())
    for key, color in zip(keys, colors):
        if max(accs_dic[key]) <= 1: 
            accs_dic[key] = np.array(accs_dic[key])*100
        x = np.arange(len(accs_dic[key]))
        plt.plot(x, accs_dic[key], f"-", label=key)
    plt.xlabel(xlabel)
    plt.ylabel("accuracy")
    plt.ylim(ylim_min, 100)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    csv_paths = (
        # 'save/220623_023400/train_step_result.csv', 
        # 'save/220624_031725/train_step_result.csv',
        'outputs/2022-06-24/12-00-33/train_step_result.csv',
        'outputs/2022-06-24/13-03-29/train_step_result.csv',
        'outputs/2022-06-24/15-50-05/train_step_result.csv',
    )
    titles = (
        # 'Drop_err',
        # 'Relu',
        '4conv_64-512c_256h',
        '5conv_64-512c_512h',
        '4conv_32-256c_256h',
    )

    losses_dict, accs_dict = {}, {}
    for csv_path, title in zip(csv_paths, titles):
        _, losses, accs, phoneme_accs = read_csv(csv_path)
        losses_dict[title] = losses
        accs_dict[title] = accs
        accs_dict[title+' (phoneme)'] = phoneme_accs

    loss_graphs(losses_dict, smooth=False, xlabel='iterations (20 times)')
    accuracy_graphs(accs_dict, xlabel='iterations (20 times)')