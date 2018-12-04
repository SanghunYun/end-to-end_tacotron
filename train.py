# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

import sys
import os
from hyperparams import Hyperparams as hp
from tqdm import tqdm
from data_load import get_batch, load_vocab
import torch

from utils import *




class Train:
    def __init__(self):
        self.char2idx, self.idx2char = load_vocab()
        self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()

        self.transcript_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, Tx, E)   ####embed_size는 input 알파벳이 embed된 사이즈, hp.vocab는 가지수, 
        self.reference_inputs = torch.expand(self.ref, -1)    ####맞는지 모르겠음

        self.decoder_inputs = torch.cat((torch.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)  #######가운데에 -1?, 앞에 concat되는건 go인가, decoder엔 y가 들어감
        #################padding 붙이는 거??###########Ty는 뭐지, r은 Tacotron에서의 3일듯###
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:]

