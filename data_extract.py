from data import HWKoDataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt 


from rich.progress import track
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_spacing_lengths(x, min_brightness=3, min_space=20):
    row_len = x.shape[2] # (C, H, (W))

    lens_space = []
    space = 0
    detected = False
    appended = False

    for now_x in range(1, row_len, 1):
        v_line = x[:, :, now_x:now_x+1]

        if torch.sum(v_line) > min_brightness: # 글씨가 감지되면
            if not detected and appended: # 새로운 감지가 시작되는 순간
                lens_space.append(space)
                appended = False
            space = 0
            detected = True

        else: # 글씨가 감지되지 않으면
            space += 1
            if space == min_space: # 공백 수가 띄어쓰기 너비를 달성하면
                appended = True
            detected = False

    return (*lens_space,)


if __name__ == '__main__':
    train_set = HWKoDataset()
    specimen_len = len(train_set)

    lens_space_of_dataset = []
    idx = 0

    for x, t in track(train_set, total=specimen_len):
        lens_space_x = get_spacing_lengths(x)
        lens_space_of_dataset.extend(lens_space_x)
        
        idx += 1
        if idx > specimen_len:
            break
    
    sns.histplot(lens_space_of_dataset, binrange=(20, 120), bins=100, kde=False)
    plt.title('Distribution of spacing lengths')
    plt.show()