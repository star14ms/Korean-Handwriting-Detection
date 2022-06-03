import collections
import torch
import io

from utils.unicode import split_syllables, join_jamos


compat_jamo = ''.join(chr(i) for i in range(12593, 12643+1))
letter = " ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + compat_jamo


label_path = 'labels/2350-common-hangul.txt'
with io.open(label_path, 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()

basic_letters = ' ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
hangul_letters = ''.join(labels)
letter_baseline = basic_letters + hangul_letters


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


converter = strLabelConverter(letter, ignore_case=False)
converter_baseline = strLabelConverter_baseline(letter_baseline)
