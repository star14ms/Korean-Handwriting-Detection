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


def loss_graphs(losses_dic, smooth=True, ylim=None, xlabel='iterations', colors=TABLEAU_COLORS):
    keys = list(losses_dic.keys())
    min_len = min([len(values) for values in losses_dic.values()])
    for key, color in zip(keys, colors):
        y = smooth_curve(losses_dic[key]) if smooth else losses_dic[key]
        x = np.arange(len(y))
        plt.plot(x, y, label=key) # f"-{color}"
    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.ylim(0, ylim)
    plt.legend(loc='upper right')
    plt.get_current_fig_manager().window.showMaximized()
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
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == '__main__':

    csv_paths = (
        # 'save/220623_023400/train_step_result.csv', 
        # 'save/220624_031725/train_step_result.csv',
        # 'outputs/2022-06-24/12-00-23/train_step_result.csv',
        # 'outputs/2022-06-24/13-03-20/train_step_result.csv',
        # 'outputs/2022-06-24/15-49-53/train_step_result.csv',
        # 'outputs/2022-06-24/22-39-06/train_step_result.csv',
        # 'outputs/2022-06-24/23-45-03/train_step_result.csv',
        # 'outputs/2022-06-25/00-05-34/train_step_result.csv',
        # 'outputs/2022-06-25/00-32-58/train_step_result.csv',
        # 'outputs/2022-06-25/02-01-25/train_step_result.csv',
        # 'outputs/2022-06-25/02-54-23/train_step_result.csv',
        'outputs/2022-06-25/03-51-52/train_step_result.csv',
        'outputs/2022-06-25/12-54-10/train_step_result.csv',
        'outputs/2022-06-25/13-18-08/train_step_result.csv',
        'outputs/2022-06-25/13-57-19/train_step_result.csv',
        'outputs/2022-06-25/14-09-18/train_step_result.csv',
    )
    titles = (
        # 'Drop_err',
        # 'Relu',
        # '4L_6789c_256h (avg)',
        # '5L_6789c_512h (avg)',
        # '4L_5678c_256h (avg)',
        # '4L_5678c_128h',
        # '4L_5678c_512h',
        # '4L_6789c_512h',
        # '5L_6789c_512h',
        # '4L_5678c_512h (sep 25b)',
        # '4L_5678c_512h (sep 25b loss)',
        '4L_5678c_256h (sep)', # MACs: 0.64 G | Params: 6.68 M
        '4L_6789c_256h', # MACs: 0.84 G | Params: 11.0 M
        '4L_6778c_256h', # MACs: 0.52 G | Params: 4.61 M
        '4L_5678c_256h', # MACs: 0.21 G | Params: 4.34 M
        '4L_5678c_256h (sep copy)', # MACs: 0.64 G | Params: 6.68 M
    )
    # 4L_6789c_256h (sep) MACs: 2.52 G | Params: 20.37 M
    # 4L_6778c_256h (sep) MACs: 1.55 G | Params: 7.49 M
    losses_dict, accs_dict = {}, {}
    for csv_path, title in zip(csv_paths, titles):
        _, losses, accs, initial_accs = read_csv(csv_path)[:4]
        losses_dict[title] = losses
        accs_dict[title+' (initial)'] = initial_accs

    loss_graphs(losses_dict, smooth=True, xlabel='iterations (20 times)')
    accuracy_graphs(accs_dict, xlabel='iterations (20 times)')