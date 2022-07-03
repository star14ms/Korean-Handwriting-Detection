import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


to_pil = ToPILImage()
to_tensor = ToTensor()


class Resize():
    def __init__(self, size=(64,64), min_crop_size=32, resample=Image.LANCZOS):
        self.size = size
        self.resample = resample
        self.min_crop_size = min_crop_size

    def __call__(self, img):
        if isinstance(img, (np.ndarray, torch.Tensor)):
           img = to_pil(img)

        arr = to_tensor(img)
        # img.show()
        crop_idxs = get_idxs_to_crop(arr, self.min_crop_size)
        img = img.crop(crop_idxs)
        # img.show()
        img = img.resize(self.size, self.resample)
        arr = to_tensor(img)
        # img.show()
    
        return arr


def get_idxs_to_crop(x, min_crop_size=32): 
    row_len = x.shape[2] # (C, H, (W))
    col_len = x.shape[1] # (C, (H), W)
    
    yb = col_len
    while torch.sum(x[:, yb-1:yb, :]) == 0 and yb-1 > 0:
        yb -= 1

    yt = 0
    while torch.sum(x[:, yt:yt+1, :]) == 0 and yt+1 < yb:
        yt += 1

    xr = row_len
    while torch.sum(x[:, :, xr-1:xr]) == 0 and xr-1 > 0:
        xr -= 1

    xl = 0
    while torch.sum(x[:, :, xl:xl+1]) == 0 and xl+1 < xr:
        xl += 1

    while xr-xl < min_crop_size or (yb-yt) - (xr-xl) > 2:
        if xl-1 >= 0:
            xl -= 1
        if xr+1 <= row_len:
            xr += 1

    while yb-yt < min_crop_size or (xr-xl) - (yb-yt) > 2:
        if yt-1 >= 0:
            yt -= 1
        if yb+1 <= col_len:
            yb += 1

    return xl, yt, xr, yb
