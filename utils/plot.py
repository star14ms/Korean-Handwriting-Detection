import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def set_font(font_path='', family=''):
    '''
        ``print(mpl.matplotlib_fname())`` 이 경로에 폰트 추가 필요
        ``print(mpl.get_cachedir())`` 캐시 지우는 경로
    '''

    if family:
        font_name = family
    else:
        font_name = fm.FontProperties(fname=font_path, size=50).get_name()
    print(font_name)
    plt.rc('font', family=font_name)


def print_font_list():
    font_list = fm.findSystemFonts() 
    font_list.sort()
    fnames = []
    for fpath in font_list:
        #폰트 파일의 경로를 사용하여 폰트 속성 객체 가져오기
        fp=fm.FontProperties(fname=fpath)
        
        # 폰트 속성을 통해 파이썬에 설정해야 하는 폰트 이름 조회 
        font_name=fp.get_name() 
        
        fnames.append(font_name)
        
    for idx, fname in enumerate(fnames):
        print(str(idx).ljust(4), fname)
     
    input()


def show_imgs_scores_softmaxes(xs, ys, title='', title_info=[], text_info=[], dark_mode=True):
    len_ys = len(ys)
    cmap='gray' if dark_mode else plt.cm.gray_r
    size = 4 if len_ys != 1 else 2
    bar1_idxs = [2, 4, 10, 12] if len_ys != 1 else [2]
    bar2_idxs = [6, 8, 14, 16] if len_ys != 1 else [4]

    n_plots = 0
    for i, x, y in zip(range(len_ys), xs, ys):
        if n_plots == 0: 
            fig = plt.figure()
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
        n_plots += 1

        plt.rcParams["font.size"] = 20
        ax = fig.add_subplot(size//2, size, 2*n_plots-1, xticks=[], yticks=[])
        if title_info!=[]:
            info = title_info[i] if title_info!='idx' else i+1
            if title=='':
                plt.title(f'{i+1}/{len_ys}\n{info}')
            else:
                plt.title(f'{i+1}/{len_ys}\n{title} ({info})') ### plot() 후에 나와야 함

        if text_info!=[]:
            ax.text(0, 0, text_info[i], ha="left", va="top", color='white')
            fig.canvas.draw()

        ax.imshow(x[0], cmap=cmap, interpolation='nearest')

        plt.rcParams["font.size"] = 11
        x = np.arange(len(y))
        ax = fig.add_subplot(size, size, bar1_idxs[n_plots-1], xticks=x, yticks=np.round(sorted(y), 1), xlabel='손글씨 숫자 예측 | 위: 점수 | 아래: 확률(%)')
        ax.bar(x, y)
        
        y = softmax(np.array(y)) * 100
        ax = fig.add_subplot(size, size, bar2_idxs[n_plots-1], xticks=x, yticks=sorted(y)[8:], ylim=(0, 100))
        ax.bar(x, y)
        
        if (i+1) % 4 == 0 or i == len_ys-1: 
            n_plots = 0
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    import matplotlib as mpl
    print_font_list()

    print(mpl.matplotlib_fname())
    print(mpl.get_cachedir())
    set_font(family='BM JUA_TTF')
    set_font(family='NanumBarunpen')
    exit()


from utils.utils import softmax
from tools import CHAR_INITIALS_PLUS, CHAR_MEDIALS_PLUS, CHAR_FINALS_PLUS


ys_plot_kwargs = [
    {
        'xticklabels': CHAR_INITIALS_PLUS,
        'xlabel': '초성',
    },
    {
        'xticklabels': CHAR_MEDIALS_PLUS,
        'xlabel': '중성',
    },
    {
        'xticklabels': CHAR_FINALS_PLUS,
        'xlabel': '종성',
    },
]


def show_img_and_scores(x, *ys, ys_kwargs=ys_plot_kwargs, title='', dark_mode=True):
    while x.shape[0] == 1:
        x = x[0]
    
    len_ys = len(ys)
    cmap='gray' if dark_mode else plt.cm.gray_r

    fig = plt.figure()
    plt.rcParams["font.size"] = 25

    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.imshow(x, cmap=cmap, interpolation='nearest')
    ax.set_title(title)

    plt.rcParams["font.size"] = 15
    for ys_idx, (y, y_kwargs) in enumerate(zip(ys, ys_kwargs)):
        sorted_idx = np.argsort(np.max(np.array(y), axis=0))
        while y.shape[0] == 1:
            y = y[0]

        color = ['black'] * len(y)
        color_order = ['purple','#1f77b4','green','orange','red']
        for y_idx, color_name in zip(range(-1, -6, -1), color_order):
            color[sorted_idx[y_idx]] = color_name

        x_ = np.arange(len(y))
        ax = fig.add_subplot(len_ys, 2, (ys_idx+1)*2, xticks=x_, yticks=np.round(sorted(y)[-3:], 1), ylabel='점수', **y_kwargs)
        ax.bar(x_, y, color=color)
         
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.pause(0.01)
    plt.show()


