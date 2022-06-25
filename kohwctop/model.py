import torch
from torch import nn

from kohwctop.save_feature_module import SaveFeatureModule
from tools import CHAR_INITIALS_PLUS, CHAR_MEDIALS_PLUS, CHAR_FINALS_PLUS


def Conv2d_Norm(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
    activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU(),
    }
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        activations[activation],
    )


def Linear_Norm(in_features, out_features, activation='relu'):
    activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU(),
    }
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        activations[activation],
    )


len_initial = len(CHAR_INITIALS_PLUS)
len_medial = len(CHAR_MEDIALS_PLUS)
len_final = len(CHAR_FINALS_PLUS)


class KoCtoPSmall(SaveFeatureModule):
    def __init__(
        self, 
        input_size=64, 
        layer_in_channels=(1, 64, 128), 
        layer_out_channels=(64, 128, 256),
        hiddens=128,
    ) -> None:
        super().__init__()
        assert len(layer_in_channels) == len(layer_out_channels)
        for next_in_chan, previous_out_chan in zip(layer_in_channels[1:], layer_out_channels[:-1]):
            assert next_in_chan == previous_out_chan
        
        c2_f_size = input_size // (2**2)
        c3_f_size = input_size // (2**3)
        in_features = (layer_out_channels[-2]*c2_f_size*c2_f_size) + (layer_out_channels[-1]*c3_f_size*c3_f_size)

        self.conv_pool_layers = nn.ModuleList()
        for in_channels, out_channels in zip(layer_in_channels, layer_out_channels):
            self.conv_pool_layers.append(
                nn.Sequential(
                    Conv2d_Norm(in_channels, out_channels), 
                    Conv2d_Norm(out_channels, out_channels), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
        
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
        )
        
        self.linear_initial = nn.Sequential(
            Linear_Norm(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_initial),
            nn.Dropout(),
        )
        self.linear_medial = nn.Sequential(
            Linear_Norm(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_medial),
            nn.Dropout(),
        )
        self.linear_final = nn.Sequential(
            Linear_Norm(in_features, hiddens),
            nn.Dropout(),
            nn.Linear(hiddens, len_final),
            nn.Dropout(),
        )

    def forward(self, x):
        super().forward(x)
        x = self.conv_pool_layers[0](x) # [N, 64, 32, 32]
        x2 = self.conv_pool_layers[1](x) # [N, 128, 16, 16]
        x3 = self.conv_pool_layers[2](x2) # [N, 256, 8, 8]
        
        if self.saving_features_available:
            self.save_feature_map('conv1', x)
            self.save_feature_map('conv2', x2)
            self.save_feature_map('conv3', x3)

        x2 = self.flatten(x2) # [N, 32768]
        x3 = self.flatten(x3) # [N, 16384]
        x = torch.cat([x2, x3], dim=1) # [N, 49152]
        del x2, x3

        y1 = self.linear_initial(x)
        y2 = self.linear_medial(x)
        y3 = self.linear_final(x)

        return y1, y2, y3


class KoCtoP(SaveFeatureModule):
    def __init__(
        self,
        input_size=64, 
        layer_in_channels=(1, 64, 128, 256), 
        layer_out_channels=(64, 128, 256, 512),
        hiddens=256,
        conv_activation='relu',
        ff_activation='relu',
    ) -> None:
        super().__init__()
        assert len(layer_in_channels) == len(layer_out_channels)
        for next_in_chan, previous_out_chan in zip(layer_in_channels[1:], layer_out_channels[:-1]):
            assert next_in_chan == previous_out_chan
        
        last_conv_feature_size = input_size // (2**len(layer_in_channels))
        in_features = (layer_out_channels[-1]*(last_conv_feature_size**2))
        
        self.conv_pool_layers = nn.ModuleList()
        for in_channels, out_channels in zip(layer_in_channels, layer_out_channels):
            self.conv_pool_layers.append(
                nn.Sequential(
                    Conv2d_Norm(in_channels, out_channels, activation=conv_activation), 
                    Conv2d_Norm(out_channels, out_channels, activation=conv_activation), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
        )
        
        self.linear_initial = nn.Sequential(
            Linear_Norm(in_features, hiddens, ff_activation),
            nn.Dropout(),
            nn.Linear(hiddens, len_initial),
            nn.Dropout(),
        )
        self.linear_medial = nn.Sequential(
            Linear_Norm(in_features, hiddens, ff_activation),
            nn.Dropout(),
            nn.Linear(hiddens, len_medial),
            nn.Dropout(),
        )
        self.linear_final = nn.Sequential(
            Linear_Norm(in_features, hiddens, ff_activation),
            nn.Dropout(),
            nn.Linear(hiddens, len_final),
            nn.Dropout(),
        )

    def forward(self, x):
        for conv_pool in self.conv_pool_layers:
            x = conv_pool(x) # [N, 64, 32, 32], [N, 128, 16, 16], [N, 256, 8, 8], [N, 512, 4, 4]
        x = self.flatten(x) # [N, 8192]

        yi = self.linear_initial(x) # 초성 (initial)
        ym = self.linear_medial(x) # 중성 (medial)
        yf = self.linear_final(x) # 종성 (final)

        return yi, ym, yf


class KoCtoP_Sep(SaveFeatureModule):
    def __init__(
        self,
        input_size=64, 
        layer_in_channels=(1, 32, 64, 128), 
        layer_out_channels=(32, 64, 128, 256),
        hiddens=256,
        conv_activation='relu',
        ff_activation='relu',
    ) -> None:
        super().__init__()
        assert len(layer_in_channels) == len(layer_out_channels)
        for next_in_chan, previous_out_chan in zip(layer_in_channels[1:], layer_out_channels[:-1]):
            assert next_in_chan == previous_out_chan
        
        last_conv_feature_size = input_size // (2**len(layer_in_channels))
        in_features = (layer_out_channels[-1]*(last_conv_feature_size**2))
        
        self.i_conv_layers_list = nn.ModuleList()
        self.m_conv_layers_list = nn.ModuleList()
        self.f_conv_layers_list = nn.ModuleList()
        
        self.imf_conv_layers_list = [self.i_conv_layers_list, self.m_conv_layers_list, self.f_conv_layers_list]
        for conv_layers in self.imf_conv_layers_list:
            for in_channels, out_channels in zip(layer_in_channels, layer_out_channels):
                conv_layers.append(
                    nn.Sequential(
                        Conv2d_Norm(in_channels, out_channels, activation=conv_activation), 
                        Conv2d_Norm(out_channels, out_channels, activation=conv_activation), 
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                )
   
        self.imf_ff_layers = nn.ModuleList() # 초성 (initial), 중성 (medial), 종성 (final)
        for output_dim in [len_initial, len_medial, len_final]:
            self.imf_ff_layers.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(),
                    Linear_Norm(in_features, hiddens, ff_activation),
                    nn.Dropout(),
                    nn.Linear(hiddens, output_dim),
                    nn.Dropout(),
                )
            )

    def forward(self, x):
        ys = []
        for idx, (conv_layers_list, ff_layers) in enumerate(zip(self.imf_conv_layers_list, self.imf_ff_layers)):
            y = conv_layers_list[0](x) # [N, 64, 32, 32]
            if self.saving_features_available:
                self.save_feature_map('conv1', y)
            y = conv_layers_list[1](y) # [N, 128, 16, 16]
            y = conv_layers_list[2](y) # [N, 256, 8, 8]
            y = conv_layers_list[3](y) # [N, 512, 4, 4]
            ys.append(ff_layers(y)) # [N, 8192]

        return (*ys,)
