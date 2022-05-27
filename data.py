import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image
from PIL.ImageOps import invert

from utils import get_file, unzip


def read_json(label_path: str):
    'json파일 불러오기'
    with open(label_path, 'r', encoding="UTF-8") as f:
        json_data = json.load(f)
    return json_data


class HWKoDataset(Dataset):
    def __init__(self, data_dir='./data/', transform=None, target_transform=None, train=True):
        self.data_dir = data_dir
        self.img_dir = f'{data_dir}train/images/' if train else f'{data_dir}test/images/'
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        if not os.path.exists(self.img_dir):
            self.prepare()

        self.data = os.listdir(self.img_dir)
        self.label = read_json(f'{data_dir}train/labels.json')['annotations'] if train else None
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(self.img_dir + self.data[idx]).convert('L')
        image = self.to_tensor(invert(image))
        label = [label for label in self.label if label["file_name"] == file_name][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample

    def prepare(self, file_name='01_[이미지이지AI]한국어_손글씨_탐지_모델'):
        file_path = f'{self.data_dir}{file_name}.zip'
        url = f'https://kr.object.ncloudstorage.com/drupal-public/nipa-playground/{file_name}.zip'
        
        if not os.path.exists(f'{self.data_dir}{file_name}.zip'):
            get_file(url, file_name=f'{file_name}.zip', cache_dir=self.data_dir)
        
        unzip(file_path, unzip_path=self.data_dir)
        print('한국어 손글씨 데이터 다운로드 완료!')


if __name__ == '__main__':
    from rich import print

    train_set = HWKoDataset()

    for x, t in train_set:
        train_set.to_pil(x).show()
        print(t)
        input()