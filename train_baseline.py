import os
import pandas as pd
import argparse
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from baseline.data import CustomDataset, alignCollate, strLabelConverter
from baseline.model import CRNN, weights_init
from baseline.common import train, test, validation, create_json
from baseline.save import load_model
from utils.rich import new_progress, console


parser = argparse.ArgumentParser()
parser.add_argument('--imgh', type=int, dest='imgh',
                        default=64,
                        help='모델에 넣기 위해 리사이징할 이미지 높이')
parser.add_argument('--imgw', type=int, dest='imgw',
                        default=1000,
                        help='모델에 넣기 위해 리사이징할 이미지 너비')
parser.add_argument('--epoch', type=int, dest='epochs',
                        default=10,
                        help='몇 바퀴 학습시킬 건지 (num epochs)')
parser.add_argument('--batch', type=int, dest='batch',
                        default=5,
                        help='묶어서 학습할 숫자 (batch size)')
parser.add_argument('--load-model', type=str, dest='load_model',
                        default=None,
                        help='save/에 저장된 불러올 모델 이름 (model weight path to load saved at save/)')
args = parser.parse_args()


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
console.log("Using [green]{}[/green] device\n".format(device))
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

# 경로 설정
DATASET_PATH = './data/'


# DATASET 만들기
train_dataset = CustomDataset(DATASET_PATH, phase='train')
validation_dataset = CustomDataset(DATASET_PATH, phase='val')

# 데이터 로드 파라미터
batch = args.batch
imgH = args.imgh
imgW = args.imgw

# DATASET 로딩하기
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=alignCollate(imgH=imgH, imgW=imgW))
val_loader = DataLoader(validation_dataset, batch_size=batch, shuffle=False, collate_fn=alignCollate(imgH=imgH, imgW=imgW))


# with open(os.path.join('data/train/labels' + '.json'), 'r', encoding="UTF-8") as label_json :
#     handwritten_json = json.load(label_json)
    
# sorted(handwritten_json['annotations'], key=lambda x: x['file_name'])[0]['text']
# handwritten_json['annotations'][:5]


# 파라미터 지정   
compat_jamo = ''.join(chr(i) for i in range(12593, 12643+1))
letter = " ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + compat_jamo
lr = 0.0001
cuda = True
start_epoch = 0
num_epochs = args.epochs
model_name = f'{num_epochs}_h{imgH}_w{imgW}' if args.load_model is None else args.load_model
mode = "train"
prediction_dir = '/prediction/'
print_iter = 500
nclass = len(letter) + 1
nc = 1
console.log('label num', len(letter))


# 모델 선언
new_model = CRNN(imgH, nc, nclass, 256)
new_model.apply(weights_init)
if args.load_model is not None:
    load_model(model_name, new_model) 
if args.load_model is not None:
    start_epoch = int(model_name.split('_')[0])
    num_epochs = start_epoch + num_epochs
    model_name = f'{num_epochs}_h{imgH}_w{imgW}'

converter = strLabelConverter(letter, ignore_case=True)
    
images = torch.FloatTensor(batch, 1, imgH, imgW)
texts = torch.IntTensor(batch * 1000)
lengths = torch.IntTensor(batch)
    
images = Variable(images)
texts = Variable(texts)
lengths = Variable(lengths)

#check parameter of model
console.log("------------------------------------------------------------")
total_params = sum(p.numel() for p in new_model.parameters())
console.log("num of parameter : ",total_params)
trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
console.log("num of trainable_ parameter :",trainable_params)
console.log("------------------------------------------------------------")


console.log('train start\n')
params = [p for p in new_model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
with new_progress() as progress:
    train(num_epochs, new_model, device, train_loader, val_loader, images, texts, lengths, converter, optimizer, lr_scheduler, prediction_dir, progress, save_desc=f'_h{imgH}_w{imgW}', start_epoch=start_epoch)


console.log('test start')
test_dataset = CustomDataset(DATASET_PATH, phase='test')
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=alignCollate(imgH=imgH, imgW=imgW))
load_model(model_name, new_model)
with new_progress() as progress:
    test_imgs, test_preds = test(new_model, device, test_loader, images, texts, lengths, converter, prediction_dir, progress)


with new_progress() as progress:
    validation(new_model, device, val_loader, images, texts, lengths, converter, prediction_dir, progress)


# 제출 결과 저장
submit = pd.DataFrame(columns = ['file_name','text'])
for i,img in enumerate(test_imgs):
    submit.loc[len(submit)] = [img.split('/')[-1], test_preds[i]]


submit.head(10)

# 제출 파일 제작
today = datetime.now().strftime('%m%d%y')
create_json(test_imgs, test_preds, file_path=f'save/pred_{today}_h{imgH}_w{imgW}.json')

# submit.to_csv(f'save/pred_{today}_h{imgH}_w{imgW}.csv', index=False, quoting=csv.QUOTE_NONE, sep='\t')
