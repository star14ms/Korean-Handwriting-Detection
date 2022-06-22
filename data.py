import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Compose
from glob import glob

from PIL import Image
from PIL.ImageOps import invert

from utils.utils import get_file, unzip, Resize
import tools as hangul_tools


def read_json(label_path: str):
    'json파일 불러오기'
    with open(label_path, 'r', encoding="UTF-8") as f:
        json_data = json.load(f)
    return json_data


class KoHWSentenceDataset(Dataset):
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    def __init__(self, data_dir='./data/', transform=Compose([ToTensor()]), target_transform=None, train=True):
        self.data_dir = data_dir
        self.img_dir = f'{data_dir}train/images/' if train else f'{data_dir}test/images/'
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.img_dir):
            self.prepare()

        self.data = sorted(glob(self.img_dir + '/*.png'))
        self.len = len(self.data)

        annotations = read_json(f'{data_dir}train/labels.json')['annotations']
        annotations = sorted(annotations, key=lambda x: x['file_name'])

        if not train:
            self.labels = ['dummy' for _ in self.data]
        else:
            self.labels = [anno['text'] for anno in annotations]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(file_name).convert('L')
        image = invert(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample

    def get_img_path(self, idx):
        return self.data[idx]

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

    def __init__(self, data_dir='./data-syllable/', transform=Compose([Resize()]), target_transform=None, train=True):
        self.data_dir = data_dir
        self.img_dir = f'{data_dir}train/' if train else f'{data_dir}test/'
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.img_dir):
            self.prepare()

        # self.data = sorted(glob(self.img_dir + '/*.jpeg'))
        self.data = tuple(os.listdir(self.img_dir))
        self.len = len(self.data)

        # annotations = read_json(f'{data_dir}label.json')['annotations']
        # annotations = sorted(annotations, key=lambda x: x['file_name'])
        # self.labels = [anno['label'] for anno in annotations]
        
        self.labels = read_json(f'{data_dir}label.json')['annotations']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(self.img_dir + file_name).convert('L')
        # label = self.labels[idx]
        label = [label['label'] for label in self.labels if label["file_name"] == file_name][0] if self.labels is not None else ''
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample

    def get_img_path(self, idx):
        return self.data[idx]

    @staticmethod
    def prepare():
        print('fonts/ 안의 폰트로 음절 데이터셋 생성 중...')

        hangul_tools.generate_hangul_images()
        hangul_tools.syllable_to_phoneme()
        hangul_tools.seperate_data_train_and_test()

        print('\n음절 데이터셋 생성 완료!')


if __name__ == '__main__':
    from rich import print

    train_set = KoSyllableDataset(transform=Resize()) # syllable: 음절
    test_set = KoSyllableDataset(train=False)
    # sentence_set = KoHWSentenceDataset()
    # sentence_set2 = KoHWSentenceDataset(train=False)

    for x, t in train_set:
        train_set.to_pil(x).show()
        print(t)
        input()