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
class TestConfig:
    batch_size: int = 50   # 묶어서 학습할 숫자 (batch size)
    print_every: int = 20  # 테스트 로그 출력하는 간격 (단위: batch))
    load_model: str = ''   # 테스트할 모델 날짜 부분 경로 (model weight date path to load)


@dataclass
class ConvNetConfig:
    input_size: int = 64                                # 모델에 넣는 이미지 크기
    layer_in_channels: List[int] = (1, 64, 128, 256)    # 합성곱 계층으로 들어가는 채널 수들
    layer_out_channels: List[int] = (64, 128, 256, 512) # 합성곱 계층에서 나오는 채널 수들
    hiddens: int = 256                                  # 완전연결 계층 노드 수
    conv_activation: str = 'relu'                       # 합성곱 계층 활성화 함수
    ff_activation: str = 'relu'                         # 완전연결 계층 활성화 함수
    dropout: float = 0.5
