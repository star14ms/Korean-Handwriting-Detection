from data import HWKoDataset
from torch.utils.data import DataLoader
from rich import print
from rich.traceback import install
install()


train_set = HWKoDataset()

for x, t in train_set:
    print(x.shape)
    print(t)
    input()