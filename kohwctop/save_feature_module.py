import torch
from torch import nn
import matplotlib.pyplot as plt
import contextlib
import math
import numpy as np


class SavingBatchError(Exception):
    def __str__(self):
        return "Too Big Input to Save Feature Maps"


class SaveFeatureModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saving_features_available = False
        self.feature_maps = {}

    @contextlib.contextmanager
    def using_config(self, name, value):
        old_value = getattr(self, name)
        setattr(self, name, value)
        try:
            yield
        finally:
            setattr(self, name, old_value)
        
    def saving_features(self):
        return self.using_config('saving_features_available', True)

    def save_feature_map(self, name, value):
        if self.saving_features_available:
            self.feature_maps[name] = torch.clone(value).cpu()

    def forward(self, x):
        if self.saving_features_available and x.shape[0] != 1:
            self.saving_features_available = False
            raise SavingBatchError()

    def imgs_show(self, img, nx='auto', filter_chans_show=False, title='', title_img=None, subtitles=[], text_info=[], dark_mode=True, full_screen=True,
        adjust={'left':0, 'right':1, 'bottom':0.02, 'top':0.98, 'hspace':0.05, 'wspace':0.02}):
        """
        c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
        """
        cmap='gray' if dark_mode else plt.cm.gray_r
    
        if len(img.shape) < 4:
            add = [1]*(4-len(img.shape))
            img = img.reshape(*add, *img.shape)
    
        FN, C, _, _ = img.shape
        imgs_num = FN if not filter_chans_show else C
        if imgs_num == 1:
            nx = 1
        elif nx == 'auto': 
            nx = math.ceil(math.sqrt(imgs_num*108/192)/108*192)
            # print(nx)
        ny = int(np.ceil(imgs_num / nx)) + 1
    
        fig = plt.figure()
        fig.subplots_adjust(**adjust)
    
        # 제목
        ax = fig.add_subplot(ny, 1, 1, xticks=[], yticks=[])
        ax.text(0.8, 0.5, f'{title}{img.shape}', size=30, ha='center', va='center')
    
        if title_img is not None:
            ax = fig.add_subplot(ny, 1, 1, xticks=[], yticks=[])
            ax.imshow(title_img[0, 0], cmap=cmap, interpolation='nearest')
    
        # 내용
        for i in range(imgs_num):
            ax = fig.add_subplot(ny, nx, nx+i+1, xticks=[], yticks=[])
            if subtitles!=[]:
                plt.title(f'\n{subtitles[i]}' if subtitles!='' else f'{i+1}')
            if text_info!=[]:
                kwargs = {
                    'ha': 'left',
                    'va': 'top',
                    'fontsize': 16,
                    'color': 'Green' if info[0] in info[-1] else 'Red'
                } 
                info = text_info[i].split(' | ')
                
                ax.text(0, 0, info[0], **kwargs)
                fig.canvas.draw()
                ax.text(18, 25, info[-1], **kwargs)
                fig.canvas.draw()
            
            im = img[i, 0] if not filter_chans_show else img[0, i]
            ax.imshow(im, cmap=cmap, interpolation='nearest')
    
        if full_screen:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.pause(0.01)
        plt.show()
    
    def show_feature_maps(self, x, device, description):
        self.eval()
        with torch.no_grad(), self.saving_features():
            x = x.reshape(1, *x.shape).to(device)
            self(x)
            
            for key, value in self.feature_maps.items():
                title = f'{description} {key} - '
                self.imgs_show(value[0], filter_chans_show=True, title=title, title_img=x.cpu())
    