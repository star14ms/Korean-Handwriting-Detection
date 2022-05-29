from tools import CHAR_INITIALS_PLUS, CHAR_MEDIALS_PLUS, CHAR_FINALS_PLUS

import torch
from torch import nn


def Conv2d_Norm_ReLU(in_chans, out_chans, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
        nn.BatchNorm2d(out_chans),
        nn.ReLU(),
    )

def Liner_Norm_ReLU(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.Dropout(),
    )


len_initial = len(CHAR_INITIALS_PLUS)
len_medial = len(CHAR_MEDIALS_PLUS)
len_final = len(CHAR_FINALS_PLUS)


class KoCtoP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        c2_f_size = 64//2//2
        c3_f_size = 64//2//2//2
        in_features = (c2*c2_f_size*c2_f_size) + (c3*c3_f_size*c3_f_size)
        hiddens = 128

        self.conv1_pool = nn.Sequential(
            Conv2d_Norm_ReLU(1, c1), 
            Conv2d_Norm_ReLU(c1, c1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2_pool = nn.Sequential(
            Conv2d_Norm_ReLU(c1, c2), 
            Conv2d_Norm_ReLU(c2, c2), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3_pool = nn.Sequential(
            Conv2d_Norm_ReLU(c2, c3), 
            Conv2d_Norm_ReLU(c3, c3), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
        )
        
        self.liner_initial = nn.Sequential(
            Liner_Norm_ReLU(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_initial),
            nn.Dropout(),
        )
        self.liner_medial = nn.Sequential(
            Liner_Norm_ReLU(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_medial),
            nn.Dropout(),
        )
        self.liner_final = nn.Sequential(
            Liner_Norm_ReLU(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_final),
            nn.Dropout(),
        )

    def forward(self, x):
        x1 = self.conv1_pool(x) # [N, 64, 32, 32]
        x2 = self.conv2_pool(x1) # [N, 128, 16, 16]
        x3 = self.conv3_pool(x2) # [N, 256, 8, 8]
        x2 = self.flatten(x2) # [N, 32768]
        x3 = self.flatten(x3) # [N, 16384]
        x = torch.cat([x2, x3], dim=1) # [N, 49152]

        y1 = self.liner_initial(x)
        y2 = self.liner_medial(x)
        y3 = self.liner_final(x)

        return y1, y2, y3
