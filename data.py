import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Compose

from PIL import Image
from PIL.ImageOps import invert

from utils.utils import get_file, unzip
import tools as hangul_tools


def read_json(label_path: str):
    'json파일 불러오기'
    with open(label_path, 'r', encoding="UTF-8") as f:
        json_data = json.load(f)
    return json_data


class KoHWSentenceDataset(Dataset):
    def __init__(self, data_dir='./data/', transform=Compose([ToTensor()]), target_transform=None, train=True):
        self.data_dir = data_dir
        self.img_dir = f'{data_dir}train/images/' if train else f'{data_dir}test/images/'
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        if not os.path.exists(self.img_dir):
            self.prepare()

        self.data = tuple(os.listdir(self.img_dir))
        self.label = read_json(f'{data_dir}train/labels.json')['annotations'] if train else None
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(self.img_dir + file_name).convert('L')
        image = invert(image)
        label = [label['text'] for label in self.label if label["file_name"] == file_name][0] if self.label is not None else ''
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


class KoSyllableDataset(Dataset):
    to_chr = hangul_tools.to_chr
    to_label = hangul_tools.to_label
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    def __init__(self, data_dir='./data-syllable/', transform=Compose([ToTensor()]), target_transform=None, train=True):
        self.data_dir = data_dir
        self.img_dir = f'{data_dir}train/' if train else f'{data_dir}test/'
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.img_dir):
            self.prepare()

        self.data = tuple(os.listdir(self.img_dir))
        self.label = read_json(f'{data_dir}label.json')['annotations']
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(self.img_dir + file_name).convert('L')
        label = [label['label'] for label in self.label if label["file_name"] == file_name][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample
    
    @staticmethod
    def prepare():
        print('fonts/ 안의 폰트로 음절 데이터셋 생성 중...')

        hangul_tools.generate_hangul_images()
        hangul_tools.syllable_to_phoneme()
        hangul_tools.seperate_data_train_and_test()

        print('\n음절 데이터셋 생성 완료!')


if __name__ == '__main__':
    from rich import print

    sentence_set = KoHWSentenceDataset()
    sentence_set2 = KoHWSentenceDataset(train=False)
    train_set = KoSyllableDataset() # syllable: 음절
    test_set = KoSyllableDataset(train=False)

    for x, t in test_set:
        test_set.to_pil(x).show()
        print(t)
        input()