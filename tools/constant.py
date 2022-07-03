import os

from utils.unicode import join_jamos, CHAR_INITIALS, CHAR_MEDIALS, CHAR_FINALS


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LEN_LABEL = 2350
FONT_SIZE = 64

# Default data paths.
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 
                                  '../fonts')
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/{}-common-hangul.txt'.format(LEN_LABEL))
DEFAULT_LABEL_CSV_FILE = os.path.join(SCRIPT_PATH,
                                  '../data/syllable/labels-map.csv')
DEFAULT_OUTPUT_DIR = './data/syllable/'


CHAR_INITIALS_PLUS = [' '] + CHAR_INITIALS
CHAR_MEDIALS_PLUS = [' '] + CHAR_MEDIALS
CHAR_FINALS_PLUS = [' '] + CHAR_FINALS


to_CtoP_char = {
    'i': dict(zip(range(len(CHAR_INITIALS_PLUS)), CHAR_INITIALS_PLUS)), # 초성
    'm': dict(zip(range(len(CHAR_MEDIALS_PLUS)), CHAR_MEDIALS_PLUS)), # 중성
    'f': dict(zip(range(len(CHAR_FINALS_PLUS)), CHAR_FINALS_PLUS)), # 종성
}
to_CtoP_label = { 
    'i': dict(zip(CHAR_INITIALS_PLUS, range(len(CHAR_INITIALS_PLUS)))), # 초성
    'm': dict(zip(CHAR_MEDIALS_PLUS, range(len(CHAR_MEDIALS_PLUS)))), # 중성
    'f': dict(zip(CHAR_FINALS_PLUS, range(len(CHAR_FINALS_PLUS)))), # 종성
}


wide_labels = list()
wide_labels = wide_labels + list(['한글 음절'])
with open(f'labels/52-alphabet.txt', 'r', encoding='utf-8') as f:
    wide_labels = wide_labels + list([line.rstrip('\n') for line in f.readlines()])
with open(f'labels/51-phoneme-hangul.txt', 'r', encoding='utf-8') as f:
    wide_labels = wide_labels + list([line.rstrip('\n') for line in f.readlines()])
with open(f'labels/10-number.txt', 'r', encoding='utf-8') as f:
    wide_labels = wide_labels + list([line.rstrip('\n') for line in f.readlines()])
with open(f'labels/31-special-character.txt', 'r', encoding='utf-8') as f:
    wide_labels = wide_labels + list([line.rstrip('\n') for line in f.readlines()])

to_char = dict(zip(range(len(wide_labels)), wide_labels))
to_label = dict(zip(wide_labels, range(len(wide_labels))))


def label_to_syllable(ti, tm, tf):
    label_yi = to_CtoP_char['i'][ti]
    label_ym = to_CtoP_char['m'][tm]
    label_yf = to_CtoP_char['f'][tf]
    return join_jamos(label_yi + label_ym + label_yf)
