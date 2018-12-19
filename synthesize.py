# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from utils import spectrogram2wav, load_spectrograms
from data_load import load_data
from scipy.io.wavfile import write
import os
import sys
from glob import glob
import numpy as np
from math import ceil


from data_load import get_Dataset, DataLoader, collate_fn, load_vocab
import torch
import torch.nn as nn
from networks import *
use_cuda = torch.cuda.is_available()

def looper(ref, start, batch_size):
    num = int(ceil(float(ref.shape[0]) / batch_size)) + 1
    tiled = np.tile(ref, (num, 1, 1))[start:start + batch_size]
    return tiled, start + batch_size % ref.shape[0]


def synthesize():
    if not os.path.exists(hp.sampledir):
        os.mkdir(hp.sampledir)

    # Load data
    texts = load_data(mode="synthesize")

    """
    # pad texts to multiple of batch_size
    texts_len = texts.shape[0]
    num_batches = int(ceil(float(texts_len) / hp.batch_size))
    padding_len = num_batches * hp.batch_size - texts_len
    texts = np.pad(
        texts, ((0, padding_len), (0, 0)), 'constant', constant_values=0
    )
    """

    # reference audio
    mels, maxlen = [], 0
    files = glob(hp.ref_audio)
    for f in files:
        _, mel, _ = load_spectrograms(f)
        mel = np.reshape(mel, (-1, hp.n_mels))
        maxlen = max(maxlen, mel.shape[0])
        mels.append(mel)

    ref = np.zeros((len(mels), maxlen, hp.n_mels), np.float32)
    for i, m in enumerate(mels):
        ref[i, :m.shape[0], :] = m


    use_cuda = torch.cuda.is_available()  

    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    # Load checkpoint
    model_path = hp.checkpoint_path + '/' + 'model_epoch_7.pth'
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        print("\n-------Model load --------\n")

    except:
        raise FileNotFoundError("\n------------Model not exists------------\n")

    # Evaluation
    model = model.eval()

    wav = generate(model, texts, ref)

    """
    path = os.path.join(hp.output_path, 'result_%d_%d.wav' % (args.restore_step, i+1))
    with open(path, 'wb') as f:
        f.write(wav)

    f.close()
    print("save wav file at step %d ..." % (i+1))
    """
def generate(model,texts,ref):

    #Variables
    if use_cuda:
        text_input = Variable(torch.from_numpy(texts).type(torch.cuda.LongTensor), volatile=True).cuda()
        mel_input = Variable(torch.from_numpy(ref).type(torch.cuda.FloatTensor), volatile=True).cuda()

    else:
        text_input = Variable(torch.from_numpy(texts).type(torch.LongTensor), volatile=True)
        mel_input = Variable(torch.from_numpy(ref).type(torch.FloatTensor), volatile=True)

    _, linear_outputs = model.forward(text_input, mel_input)

    linear_outputs = linear_outputs.data.cpu().numpy()
    for i, output in enumerate(linear_outputs):
            
        print("File {}.wav is being generated ...".format(i))
        audio = spectrogram2wav(output)
        write(
            os.path.join(hp.sampledir, '{}.wav'.format(i)),
            hp.sr, audio
        )



if __name__ == '__main__':
    synthesize()
    print("Done")
