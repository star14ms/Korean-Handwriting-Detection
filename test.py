
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from thop import profile

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich.traceback import install
install()

from data import KoSyllableDataset
from data import KoHWSentenceDataset
from kohwctop import TestConfig, KoCtoPConfig, ConvNetConfig
from kohwctop.model import KoCtoP
from kohwctop.test import test, test_sample
from utils.rich import new_progress, console
from utils.plot import set_font


cs = ConfigStore.instance()
cs.store(group="test", name="test", node=TestConfig, package="test")
cs.store(group="model", name="koCtoP", node=KoCtoPConfig, package="model")
cs.store(group="model", name="convNet", node=ConvNetConfig, package="model")


@hydra.main(config_path=os.path.join('.', "configs"), config_name="test", version_base=None)
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    console.log(OmegaConf.to_yaml(config))

    set_font('NanumGothicExtraBold.ttf')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.log("Using [green]{}[/green] device\n".format(device))
    
    batch_size = config.test.batch_size
    test_set = KoSyllableDataset(train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    # sentence_set = KoHWSentenceDataset()
    
    model = KoCtoP(**config.model).to(device)
    model.load_state_dict(torch.load(config.test.load_model))
    macs, params = profile(model, inputs=(torch.randn(1, 1, 64, 64).to(device),), verbose=False)
    console.log('모델 로드 완료! (MACs: {} G | Params: {} M)'.format(
        round(macs/1000/1000/1000, 2), 
        round(params/1000/1000, 2),
    ))

    # test_sample(test_set, model, device, random_sample=True, plot_when_wrong=False, plot_feature_map=True)

    loss_fn = nn.CrossEntropyLoss()
    with new_progress() as progress:
        test(
            model, test_loader, loss_fn, 
            progress, config.test.print_every, show_wrong_info=False
        )


if __name__ == '__main__':
    main()