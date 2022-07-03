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


len_initials = len(CHAR_INITIALS_PLUS)
len_medials = len(CHAR_MEDIALS_PLUS)
len_finals = len(CHAR_FINALS_PLUS)


class KoCtoPSmall(SaveFeatureModule):
    def __init__(
        self, 
        input_size=64, 
        layer_in_channels=(1, 64, 128),
        layer_out_channels=(64, 128, 256),
        hiddens=128,
        conv_activation='relu',
        ff_activation='relu',
        dropout=0.5
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
                    Conv2d_Norm(in_channels, out_channels, activation=conv_activation), 
                    Conv2d_Norm(out_channels, out_channels, activation=conv_activation), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
        
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        
        # 초성 (initial), 중성 (medial), 종성 (final)
        output_dims = (len_initials, len_medials, len_finals)
        
        self.imf_ff_layers = nn.ModuleList()
        for output_dim in output_dims:
            self.imf_ff_layers.append(nn.Sequential(
                    Linear_Norm(in_features, hiddens, ff_activation),
                    nn.Dropout(p=dropout),
                    nn.Linear(hiddens, output_dim),
                    nn.Dropout(p=dropout),
                )
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

        yi = self.imf_ff_layers[0](x) # 초성 (initial)
        ym = self.imf_ff_layers[1](x) # 중성 (medial)
        yf = self.imf_ff_layers[2](x) # 종성 (final)

        return yi, ym, yf


class KoCtoP_Merged(SaveFeatureModule):
    def __init__(
        self,
        input_size=64, 
        layer_in_channels=(1, 32, 64, 128),
        layer_out_channels=(32, 64, 128, 256),
        hiddens=256,
        conv_activation='relu',
        ff_activation='relu',
        dropout=0.5
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
            nn.Dropout(p=dropout),
        )
        
        # 초성 (initial), 중성 (medial), 종성 (final)
        output_dims = (len_initials, len_medials, len_finals)
        
        self.imf_ff_layers = nn.ModuleList()
        for output_dim in output_dims:
            self.imf_ff_layers.append(nn.Sequential(
                    Linear_Norm(in_features, hiddens, ff_activation),
                    nn.Dropout(p=dropout),
                    nn.Linear(hiddens, output_dim),
                    nn.Dropout(p=dropout),
                )
            )

    def forward(self, x):
        for conv_pool in self.conv_pool_layers:
            x = conv_pool(x) # [N, 32, 32, 32], [N, 64, 16, 16], [N, 128, 8, 8], [N, 256, 4, 4]
        x = self.flatten(x) # [N, 4096]

        yi = self.imf_ff_layers[0](x) # 초성 (initial)
        ym = self.imf_ff_layers[1](x) # 중성 (medial)
        yf = self.imf_ff_layers[2](x) # 종성 (final)

        return yi, ym, yf


class KoCtoP(SaveFeatureModule):
    def __init__(
        self,
        input_size=64, 
        layer_in_channels=(1, 32, 64, 128),
        layer_out_channels=(32, 64, 128, 256),
        hiddens=256,
        conv_activation='relu',
        ff_activation='relu',
        dropout=0.5
    ) -> None:
        super().__init__()
        assert len(layer_in_channels) == len(layer_out_channels)
        for next_in_chan, previous_out_chan in zip(layer_in_channels[1:], layer_out_channels[:-1]):
            assert next_in_chan == previous_out_chan
        
        last_conv_feature_size = input_size // (2**len(layer_in_channels))
        in_features = (layer_out_channels[-1]*(last_conv_feature_size**2))
        
        self.i_conv_layers_list = nn.ModuleList() # 초성 (initial)
        self.m_conv_layers_list = nn.ModuleList() # 중성 (medial)
        self.f_conv_layers_list = nn.ModuleList() # 종성 (final)
        
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
   
        # 초성 (initial), 중성 (medial), 종성 (final)
        output_dims = (len_initials, len_medials, len_finals)

        self.imf_ff_layers = nn.ModuleList()
        for output_dim in output_dims:
            self.imf_ff_layers.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=dropout),
                    Linear_Norm(in_features, hiddens, ff_activation),
                    nn.Dropout(p=dropout),
                    nn.Linear(hiddens, output_dim),
                    nn.Dropout(p=dropout),
                )
            )

    def forward(self, x):
        ys = []
        for idx, (conv_layers_list, ff_layers) in enumerate(zip(self.imf_conv_layers_list, self.imf_ff_layers)):
            y = conv_layers_list[0](x) # [N, 32, 32, 32]
            if self.saving_features_available:
                self.save_feature_map('conv1', y)
            y = conv_layers_list[1](y) # [N, 64, 16, 16]
            y = conv_layers_list[2](y) # [N, 128, 8, 8]
            y = conv_layers_list[3](y) # [N, 256, 4, 4]
            ys.append(ff_layers(y)) # [N, 4096]

        return (*ys,)


class ConvNet(SaveFeatureModule):
    def __init__(
        self,
        input_size=64, 
        output_dim=1+52+51+10+31,
        layer_in_channels=(1, 32, 64, 128),
        layer_out_channels=(32, 64, 128, 256),
        hiddens=256,
        conv_activation='relu',
        ff_activation='relu',
        dropout=0.5
    ) -> None:
        super().__init__()
        assert len(layer_in_channels) == len(layer_out_channels)
        for next_in_chan, previous_out_chan in zip(layer_in_channels[1:], layer_out_channels[:-1]):
            assert next_in_chan == previous_out_chan
        
        last_conv_feature_size = input_size // (2**len(layer_in_channels))
        in_features = (layer_out_channels[-1]*(last_conv_feature_size**2))
        
        self.conv_layers_list = nn.ModuleList()
        
        for in_channels, out_channels in zip(layer_in_channels, layer_out_channels):
            self.conv_layers_list.append(
                nn.Sequential(
                    Conv2d_Norm(in_channels, out_channels, activation=conv_activation), 
                    Conv2d_Norm(out_channels, out_channels, activation=conv_activation), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
   
        self.ff_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            Linear_Norm(in_features, hiddens, ff_activation),
            nn.Dropout(p=dropout),
            nn.Linear(hiddens, output_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        y = self.conv_layers_list[0](x) # [N, 32, 32, 32]
        y = self.conv_layers_list[1](y) # [N, 64, 16, 16]
        y = self.conv_layers_list[2](y) # [N, 128, 8, 8]
        y = self.conv_layers_list[3](y) # [N, 256, 4, 4]
        y = self.ff_layers(y) # [N, 4096]

        return y