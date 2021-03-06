#!/usr/bin/env python

import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.constant import DEFAULT_FONTS_DIR, DEFAULT_LABEL_FILE, DEFAULT_OUTPUT_DIR, FONT_SIZE
from utils.rich import new_progress


# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def generate_hangul_images(
    label_file=DEFAULT_LABEL_FILE, 
    fonts_dir=DEFAULT_FONTS_DIR, 
    output_dir=DEFAULT_OUTPUT_DIR, 
    distortion_count=3
):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    
    if 'syllable' in label_file:
        label_file = './labels/2350-common-hangul.txt'
        output_dir = './data/syllable/'
    elif 'alphabet' in label_file:
        label_file = './labels/52-alphabet.txt'
        output_dir = './data/alphabet/'
    elif 'number' in label_file:
        label_file = './labels/10-number.txt'
        output_dir = './data/number/'
    elif 'phoneme' in label_file:
        label_file = './labels/51-phoneme-hangul.txt'
        output_dir = './data/phoneme/'
    elif 'special' in label_file:
        label_file = './labels/31-special-character.txt'
        output_dir = './data/special/'


    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')

    progress = new_progress()
    progress.start()
    n_label = int(label_file.split('/')[-1].split('-')[0])
    total = n_label * len(fonts) * (distortion_count + 1)
    task_id = progress.add_task('[red]generate images', total=total)

    total_count = 0
    prev_count = 0
    for character in labels:

        if character in '",\\':
            character = '\\' + character

        # Print image count roughly every 5000 images.
        if total_count - prev_count >= 5000:
            prev_count = total_count
            progress.log('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            font = ImageFont.truetype(font, FONT_SIZE)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill=(255),
                font=font
            )
            file_string = '{:08d}.jpeg'.format(total_count)
            file_path = f'{image_dir}/{file_string}'
            image.save(file_path, 'JPEG')
            labels_csv.write(u'{},{}\n'.format(file_path, character))
            progress.advance(task_id)

            for i in range(distortion_count):
                total_count += 1
                file_string = '{:08d}.jpeg'.format(total_count)
                file_path = f'{image_dir}/{file_string}'
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'JPEG')
                labels_csv.write(u'{},{}\n'.format(file_path, character))
                progress.advance(task_id)

    progress.log('Finished generating {} images.'.format(total_count))
    labels_csv.close()
    progress.stop()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    parser.add_argument('--distortion-count', type=str, dest='distortion_count',
                        default=3,
                        help='Number of random distortion images to generate '
                             'per font and character.')                      
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir, args.distortion_count)
