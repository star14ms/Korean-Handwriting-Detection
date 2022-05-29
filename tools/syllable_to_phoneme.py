import argparse
import os
import csv
import json

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.rich import new_progress
from utils.unicode import split_syllable_char
from tools.constant import DEFAULT_LABEL_CSV_FILE, DEFAULT_OUTPUT_DIR, to_label


def syllable_to_phoneme(label_csv_file=DEFAULT_LABEL_CSV_FILE, output_dir=DEFAULT_OUTPUT_DIR):
    with open(label_csv_file, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        total = sum(1 for _ in rdr)

    with open(label_csv_file, 'r', encoding='utf-8') as f, new_progress() as progress:
        rdr = csv.reader(f)
        task_id = progress.add_task('[yellow]syllable to phoneme', total=total)

        label_phonemes = []
        for line in rdr:
            file_name = line[0]
            label = line[1]

            phoneme = list(split_syllable_char(label))
            phoneme[0] = phoneme[0] if phoneme[0] is not None else ' '
            phoneme[1] = phoneme[1] if phoneme[1] is not None else ' '
            phoneme[2] = phoneme[2] if phoneme[2] is not None else ' '
            new_label = {
                'file_name': file_name,
                'label_full': label,
                'label': {
                    'initial': to_label['i'][phoneme[0]],
                    'medial': to_label['m'][phoneme[1]],
                    'final': to_label['f'][phoneme[2]],
                },
                'phoneme': {
                    'initial': phoneme[0],
                    'medial': phoneme[1],
                    'final': phoneme[2],
                },
            }
            label_phonemes.append(new_label)
            progress.advance(task_id)


        file_path = output_dir + 'label.json'
        label_json = {'annotations': label_phonemes}

        with open(file_path, 'w', encoding='utf-8') as f_out:
            json.dump(label_json, f_out, indent=4, ensure_ascii = False)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-csv-file', type=str, dest='label_csv_file',
                        default=DEFAULT_LABEL_CSV_FILE,
                        help='Generated csv file after run hangul-image-generator.py')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store label JSON file.')
    args = parser.parse_args()
    
    syllable_to_phoneme(args.label_csv_file, args.output_dir)

