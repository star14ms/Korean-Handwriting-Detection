import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from thop import profile

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich.traceback import install
install()

from data import KoSyllableDataset
from kohwctop.model import KoCtoP
from kohwctop import TrainConfig, ConvNetConfig
from kohwctop.trainer import Trainer
from utils.rich import console


def train(config: DictConfig, save_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.log("Using [green]{}[/green] device\n".format(device))

    train_set = KoSyllableDataset()
    test_set = KoSyllableDataset(train=False)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=True)
    console.log(f'데이터 로드 완료! (train set: {len(train_set)} / test set: {len(test_set)})')
    
    model = KoCtoP(**config.model).to(device)
    macs, params = profile(model, inputs=(torch.randn(1, 1, 64, 64).to(device),), verbose=False)
    
    model = KoCtoP(**config.model).to(device)
    console.log('모델 생성 완료! (MACs: {} G | Params: {} M)'.format(
        round(macs/1000/1000/1000, 2), 
        round(params/1000/1000, 2),
    ))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    
    trainer = Trainer(
        model, train_loader, test_loader, loss_fn, optimizer, device, 
        config.train.print_every, save_dir, config.train.load_model,
    )
    trainer.train(config.train.epoch)


cs = ConfigStore.instance()
cs.store(group="train", name="train", node=TrainConfig, package="train")
cs.store(group="model", name="convNet", node=ConvNetConfig, package="model")


@hydra.main(config_path=os.path.join('.', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    console.log(OmegaConf.to_yaml(config))

    if config.train.load_model:
        save_datetime = config.train.load_model
    else:
        ymd = os.listdir('./outputs')[-1]
        hms = os.listdir('./outputs/'+ymd)[-1]
        save_datetime = f'{ymd}/{hms}'
        
    save_dir = f'outputs/{save_datetime}/'

    train(config, save_dir)


if __name__ == '__main__':
    main()