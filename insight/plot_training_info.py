import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv(csv_path, return_dict=False):
    df = pd.read_csv(csv_path)
    
    col_list = []
    for col in df.keys():
        col_list.append(df[col].tolist())

    if return_dict:
        return dict(zip(*col_list))
    else:
        return (*col_list,)


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def loss_graph(losses, smooth=True, ylim=None):
    x = np.arange(len(losses))
    y = smooth_curve(losses) if smooth and len(losses) > 10 else losses
    plt.plot(x, y, f"-", label="loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, ylim)
    plt.legend(loc='upper right')
    plt.show()


def accuracy_graph(accs, ylim_min=None):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(accs))
    if max(accs) <= 1: 
        accs = np.array(accs)*100
    plt.plot(x, accs, f"-", label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(ylim_min, 100)
    plt.legend(loc='lower right') # 그래프 이름 표시
    plt.show()


if __name__ == '__main__':
    csv_path = 'save/train_step_result.csv'
    _, losses, accs = read_csv(csv_path)

    loss_graph(losses, smooth=False)
    accuracy_graph(accs)