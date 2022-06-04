import os
import json
import collections
from PIL import Image
from glob import glob
import io

import torch
from torch.utils import data
import torchvision.transforms as transforms

from utils.unicode import split_syllables, join_jamos


class CustomDataset(data.Dataset):
    """ 필수 함수 : 
        - __init__ : 초기화
        - __len__ : 데이터셋(input)의 길이 반환
        - __getitem__ : 데이터셋을 인덱스로 불러옴
        
        그 외 함수:
        - get_root : 경로 반환
        - get_img_path : 인덱스 출력
    """
 
    def __init__(self, root, phase='train', transform=None, target_transform=None):
        # 경로 생성 후 생성된 경로의 이미지 파일을 불러와 정렬한 다음 저장
        self.root = os.path.join(root, phase)
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        annotations = None

        if phase == 'val':
            self.root = os.path.join(root, 'train')
        # 라벨 데이터인 json 파일을 불러와 저장한 다음 json 파일 안의 딕셔너리를 파일 이름 순으로 정렬
        with open(os.path.join(self.root, 'labels.json'), 'r', encoding="UTF-8") as label_json :
            label_json = json.load(label_json)
            annotations = label_json['annotations']
        annotations = sorted(annotations, key=lambda x: x['file_name'])
        
        self.imgs = sorted(glob(self.root + '/images' + '/*.png'))
        
        
        if phase == 'train':
            annotations = annotations[:int(0.9*len(annotations))]
            self.imgs = self.imgs[:int(0.9*len(self.imgs))]
        elif phase == 'val':
            annotations = annotations[int(0.9*len(annotations)):]
            self.imgs = self.imgs[int(0.9*len(self.imgs)):]

            
        for anno in annotations :
            if phase == 'test' :
                self.labels.append('dummy')
            else :
                self.labels.append(anno['text'])
        
        

    # training set의 손글씨 이미지들의 갯수 출력
    def __len__(self):
        return len(self.imgs)

    # 데이터 셋의 idx 번째 샘플 데이터를 반환
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.imgs[index]
        # 이미지 모드 변경. 흰 배경에 검은 글씨 뿐이므로 그레이 스케일('L') 지정
        img = Image.open(img_path).convert('L')
        
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)
    
    # CustomDataset 클래스의 __init__ 메서드에서 정의한 self.root 출력
    def get_root(self) :
        return self.root

    # 해당 index의 이미지 파일의 경로 출력
    def get_img_path(self, index) :
        return self.imgs[index]
    

# 이미지 사이즈 변경(resize), 이중선형보간(bilinear interpolation), 텐서 변환, 표준화(normalize) 
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

    
class alignCollate(object): 

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels = zip(*batch)
       
        imgH = self.imgH
        imgW = self.imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


class strLabelConverter(object):
  
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case        
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        
        self.dict = {}
        
        # 식별의 대상이 되는 특수문자, 숫자, 알파벳 대소문자, 한글 각 기호/글자에 넘버링
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1 
            
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str): 
            text = split_syllables(text)
            # 필요 시 영어 문자를 모두 소문자 형식으로 반환
            text = [
                self.dict[char.lower() if self._ignore_case and char.isalpha() else char]
                for char in text
            ]
            length = [len(text)]
            
        elif isinstance(text, collections.abc.Iterable):
            length = [len(split_syllables(s)) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
        # torch.numel(input) -> int : returns the total number of elements in the input tensor.
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return join_jamos(''.join([self.alphabet[i - 1] for i in t]))
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return join_jamos(''.join(char_list))
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
       
        
class strLabelConverter_baseline(object):
  
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case        
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        
        self.dict = {}
        
        # 식별의 대상이 되는 특수문자, 숫자, 알파벳 대소문자, 한글 각 기호/글자에 넘버링
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1 
            
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str): 
            # 필요 시 영어 문자를 모두 소문자 형식으로 반환
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
            
        elif isinstance(text, collections.abc.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
        # torch.numel(input) -> int : returns the total number of elements in the input tensor.
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def loadData(v, data):
    d_size = data.size()
    v.resize_(d_size).copy_(data)


# compat_jamo = ''.join(chr(i) for i in range(12593, 12643+1))
# letter = " ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + compat_jamo


# label_path = 'labels/2350-common-hangul.txt'
# with io.open(label_path, 'r', encoding='utf-8') as f:
#     labels = f.read().splitlines()

# basic_letters = ' ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
# hangul_letters = ''.join(labels)
# letter_baseline = basic_letters + hangul_letters


# converter = strLabelConverter(letter, ignore_case=False)
# converter_baseline = strLabelConverter_baseline(letter_baseline)
