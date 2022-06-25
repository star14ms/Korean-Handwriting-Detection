import matplotlib.pyplot as plt
from matplotlib._color_data import TABLEAU_COLORS
import numpy as np

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from utils.utils import read_csv


TABLEAU_COLORS = TABLEAU_COLORS.values()


def smooth_curve(x, beta=2):
    """ kaiser window smoothing """

    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, beta)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def loss_graphs(losses_dic, smooth=True, ylim=None, xlabel='Iterations', colors=TABLEAU_COLORS):
    plt.rc('font', size=18)
    keys = list(losses_dic.keys())
    min_len = min([len(values) for values in losses_dic.values()])
    for key, color in zip(keys, colors):
        y = smooth_curve(losses_dic[key]) if smooth else losses_dic[key]
        x = np.arange(len(y))
        plt.plot(x, y, label=key) # f"-{color}"
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.ylim(0, ylim)
    plt.legend(loc='upper right')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


def accuracy_graphs(accs_dic, ylim_min=None, xlabel='Epochs', colors=TABLEAU_COLORS):
    plt.rc('font', size=18)
    markers = {'train': 'o', 'test': 's'}
    keys = list(accs_dic.keys())
    for key, color in zip(keys, colors):
        if max(accs_dic[key]) <= 1: 
            accs_dic[key] = np.array(accs_dic[key])*100
        x = np.arange(len(accs_dic[key]))
        plt.plot(x, accs_dic[key], f"-", label=key)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim(ylim_min, 100)
    plt.legend(loc='lower right')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == '__main__':

    csv_paths = (
        'outputs/2022-06-25/19-44-31/train_step_result.csv',
    )
    titles = (
        '4L_5678c_256h (drop 0)', # MACs: 0.64 G | Params: 6.68 M
    )
    losses_dict, accs_dict = {}, {}
    for csv_path, title in zip(csv_paths, titles):
        _, losses, accs, initial_accs = read_csv(csv_path)[:4]
        losses_dict[title] = losses
        accs_dict[title+' (initial)'] = initial_accs

    loss_graphs(losses_dict, smooth=True, xlabel='iterations (20 times)')
    accuracy_graphs(accs_dict, xlabel='iterations (20 times)')