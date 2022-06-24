from dataclasses import dataclass
from typing import List


@dataclass
class TrainConfig:
    epoch: int = 1         # 몇 바퀴 학습시킬 건지 (num epochs)
    batch_size: int = 50   # 묶어서 학습할 숫자 (batch size)
    lr: float = 1e-3       # 학습률 (Learning Rate)
    print_every: int = 20  # 학습 로그 출력하는 간격 (단위: batch))
    load_model: str = ''   # 이어서 학습시킬 모델 날짜 부분 경로 (model weight date path to load)
    

@dataclass
class ModelConfig:
    input_size: int = 64 
    layer_in_channels: List[int] = (1, 64, 128, 256)
    layer_out_channels: List[int] = (64, 128, 256, 512)
    hiddens: int = 256
    conv_activation: str = 'relu'
    ff_activation: str = 'relu'