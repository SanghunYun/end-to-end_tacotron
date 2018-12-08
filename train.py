from data_load import get_Dataset, DataLoader, collate_fn, load_vocab
from torch import optim

import numpy as np
import os
import time
import torch
import torch.nn as nn

from modules import *
from utils import * 
from networks import *
# use_cuda = torch.cuda.is_availabe()

def main():

    # Get Dataset
    data_path = 'C:/2018-2/LJSpeech-1.1'
    dataset = get_Dataset(os.path.join(data_path,'metadata.csv'), os.path.join(data_path,'wavs'))

    # TODO: Tacotron 최종 모델 완성시키기
    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()


    # TODO: checkpoint (parameter) 불러와서 사용
    """
    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored at step %d--------\n" % args.restore_step)

    except:
        print("\n--------Start New Training--------\n")
    """    
    
    
    # Train
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)



    # Decide loss function
    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    
    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sr * 0.5) * hp.num_freq)


    for epoch in range(hp.epochs):
        
        """
         x : Texts (N, Tx)
         y : Reduced melspectrogram (N, Ty//r, n_mels*r)
         z : Magnitude (N, Ty, n_fft//2+1)
         self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
        """

        dataloader = DataLoader(dataset, batch_size=hp.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    
        for i, data in enumerate(dataloader):
            fname, text, mel, mag = data

            """
                    elif mode=='eval'
                    else # synthesize
            """

            mel_input = np.concatenate((np.zeros([hp.batch_size, hp.n_mels, 1], dtype=np.float32), mel[:,:,1:]), axis=2)
 
            
            # input : text_input, mel_input
            # for loss mel, linear
            if use_cuda:
                text_input = Variable(torch.from_numpy(text).type(torch.cuda.LongTensor), requires_grad=False).cuda()
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                mel_spectrogram = Variable(torch.from_numpy(mel).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                linear_spectrogram = Variable(torch.from_numpy(mag).type(torch.cuda.FloatTensor), requires_grad=False).cuda()

            else:
                text_input = Variable(torch.from_numpy(text).type(torch.LongTensor), requires_grad=False)
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.FloatTensor), requires_grad=False)
                mel_spectrogram = Variable(torch.from_numpy(mel).type(torch.FloatTensor), requires_grad=False)
                linear_spectrogram = Variable(torch.from_numpy(mag).type(torch.FloatTensor), requires_grad=False)

            # Forward
            mel_output, linear_output = model.forward(text_input, mel_input)

            # Loss mel, mag 
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output - linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:, :n_priority_freq, :])
            loss = mel_loss + linear_loss
            loss = loss.cuda()
            # TODO: loss.cuda()??????

            start_time = time.time()

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()

            current_step = i + epoch * len(dataloader) + 1
            time_per_step = time.time() - start_time

            if current_step % hp.log_step == 0:
                print('time per step : %.2f sec' % time_per_step)
                print('at timestpe %d' % current_step)
                print('linear loss : %.4f' % linear_loss.data[0])
                print('mel loss : %.4f' % mel_loss.data[0])
                print('total loss : %.4f' % loss.data[0])

            if current_step % hp.save_step == 0:
                save_checkpoint({'model' : model.state_dict(),
                                'optimizer' : optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print('save model at step %d ...' % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer