from data import KoSyllableDataset
from data import KoHWSentenceDataset
from kohwctop.model import KoCtoP

import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

from utils.rich import new_progress, console
from utils.plot import set_font
from kohwctop.test import test, test_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, dest='load_model',
                            default='save/220623_101400/model.pt',
                            help='불러올 모델 경로 (model weight path to load)')
    parser.add_argument('--batch-size', type=int, dest='batch_size',
                            default=50,
                            help='묶어서 테스트할 숫자 (batch size)')
    args = parser.parse_args()

    set_font(family='BM JUA_TTF')

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.log("Using [green]{}[/green] device\n".format(device))
    
    batch_size = args.batch_size
    test_set = KoSyllableDataset(train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    sentence_set = KoHWSentenceDataset()
    
    model = KoCtoP().to(device)
    model.load_state_dict(torch.load(args.load_model))
    console.log('모델 로드 완료!')

    # test_sample(test_set, model, device, random_sample=False, plot_when_wrong=False, plot_feature_map=False)

    loss_fn = nn.CrossEntropyLoss()
    with new_progress() as progress:
        test(model, test_loader, loss_fn, progress, show_wrong_info=False)
