import os
from utils import get_file, unzip


file_name = '01_[이미지이지AI]한국어_손글씨_탐지_모델'
file_path = f'data/{file_name}.zip'
url = f'https://kr.object.ncloudstorage.com/drupal-public/nipa-playground/{file_name}.zip'


if not os.path.exists(f'data/{file_name}.zip'):
    data_path = get_file(url, file_name=f'{file_name}.zip')

unzip(file_path)
print('한국어 손글씨 데이터 다운로드 완료!')