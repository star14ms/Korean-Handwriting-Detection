import os
import random
import math
import argparse

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import makedirs
from utils.rich import new_progress
import shutil
from tools.constant import DEFAULT_OUTPUT_DIR


def seperate_data_train_and_test(data_dir=DEFAULT_OUTPUT_DIR, test_rate=0.2):
    src_dir = f'{data_dir}train/'
    target_dir = f'{data_dir}test/'
    data_all = os.listdir(src_dir)

    test_data = random.sample(data_all, math.ceil(len(data_all)*test_rate))

    makedirs(target_dir)
    
    with new_progress() as progress:
        task_id = progress.add_task('[green]seperate test data', total=len(test_data))
        for file_name in test_data:
            shutil.move(src_dir+file_name, target_dir+file_name)
            progress.advance(task_id)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    seperate_data_train_and_test(args.output_dir)