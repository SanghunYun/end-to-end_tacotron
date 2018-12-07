from hyperparams import Hyperparams as hp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import *
import re
import codecs
import os
import unicodedata



def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode == "train":
        # Parse
        fpaths, texts = [], []
        transcript = os.path.join(hp.data, 'metadata.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()

        for line in lines:
            fname, _, text = line.strip().split("|")
            fpath = os.path.join(hp.data, "wavs", fname + ".wav")

            fpaths.append(fpath)

            text = text_normalize(text) + "E" # ␃: EOS
            text = [char2idx[char] for char in text]
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, texts
    else:
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + u"␃" for line in lines]  # text normalization, E: EOS
        texts = np.zeros((len(lines), hp.Tx), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts


class get_Dataset(Dataset):
    def __init__(self, csv_file, wav_file):
        
        self.metadata = pd.read_csv(csv_file, sep='|', header=None)
        self.wav_file = wav_file
    
    def __len__(self):
        return len(self.metadata)

    def load_vocab(self):
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz''' # ␀: Padding ␃: End of Sentence

        char2idx = {char: idx for idx, char in enumerate(vocab)}
        idx2char = {idx: char for idx, char in enumerate(vocab)}
        return char2idx, idx2char

    def text_normalize(self, text):
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz''' # ␀: Padding ␃: End of Sentence

        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

        text = re.sub("[^{}]".format(vocab), " ", text)
        text = re.sub("[ ]+", " ", text)
        return text

    def load_wav(self, filename):
        return librosa.load(filename, sr=22050)

    def __getitem__(self, idx):
        char2idx, idx2char = self.load_vocab()
                
        wav_name = os.path.join(self.wav_file, self.metadata.ix[idx,0]) + '.wav'
        text = self.metadata.ix[idx, 1]
        text = self.text_normalize(text) + "E"
        text = [char2idx[char] for char in text]

        text = np.asarray(text, dtype=np.int32)
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)

        return wav_name, text, wav


def collate_fn(data):
    fpath = data[0]
    text = data[1]
    wav = data[2]

    text = _prepare_data(text).astype(np.int32)
    wav = _prepare_data(wav)

    fname, mel, mag = load_spectrograms(fpath)

    return text, mag, mel

def _pad_data(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

