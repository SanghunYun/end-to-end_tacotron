from data_load import get_Dataset, DataLoader, collate_fn, load_vocab
from torch import optim

import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
from modules import *
from utils import * 
from networks import *
use_cuda = torch.cuda.is_available()


def save_checkpoint(state, filename='checkpoint.pth'):
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

def main():

    # Get Dataset
    
    dataset = get_Dataset(os.path.join(hp.data,'metadata.csv'), os.path.join(hp.data,'wavs'))

    # TODO: Tacotron 최종 모델 완성시키기
    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    

    # Train
    model = model.train()
    
    # Load checkpoint
    model_path = hp.checkpoint_path + '/' + 'model_epoch_8.pth'
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print("\n-------Model load --------\n")


    """
    number = 0
    model_out_path = hp.checkpoint_path + '/' + 'model_epoch_{}.pth'.format(number)
    model.load_state_dict(torch.load(model_out_path))

    print("\n--------load model model_epoch_{}.pth-------\n".format(number))
    """
    
    print("\n--------Start New Training--------\n")
    
    
    
   

    
    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)
    
    if not os.path.exists(hp.log_data):
        os.mkdir(hp.log_data)
    


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
                                shuffle=True, collate_fn=collate_fn, drop_last=True)
    
        for i, data in enumerate(dataloader):
            fname, text, mel, mag = data
            
            """
                    elif mode=='eval'
                    else # synthesize
            """
            #mel_input = np.concatenate((np.zeros([hp.batch_size, hp.n_mels, 1], dtype=np.float32), mel[:,:,1:]), axis=2)
            mel_input = mel.reshape(hp.batch_size, -1, hp.n_mels)
            # input : text_input, mel_input
            # for loss mel, linear
            optimizer.zero_grad()
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
            # TODO: loss.cuda()?????

            start_time = time.time()

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()

            current_step = i + epoch * len(dataloader) + 1
            time_per_step = time.time() - start_time
            
            print('(epoch, batch) (%d, %d) \n(time, current, linear, mel, loss) (%.2f  %d  %.4f  %.4f  %.4f)' % (epoch+1, i+1, time_per_step, current_step, linear_loss, mel_loss, loss))
            
            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

        print('model_epoch_{} save'.format(epoch+1))
        """
        log_path = hp.log_data + '/' + 'model_log_{}.csv'.format(epoch+1)
        df = pd.DataFrame([[epoch+1, i+1, time_per_step, current_step, linear_loss, mel_loss, loss]])
        df.to_csv(log_path, header=True, index=False)        
        """
        """
        df = pd.DataFrame([[epoch+1, i+1, time_per_step, current_step, linear_loss, mel_loss, loss]])
        if epoch != 0:
            add_df = pd.read_csv(log_path)
            df = pd.concat([add_df, df])
        df.to_csv(log_path, header=True, index=False)
        """
        """
        model_out_path = hp.checkpoint_path + '/' + 'model_epoch_{}.pth'.format(epoch+1)
        torch.save(model.state_dict(), model_out_path)
        """
    # Make trained_model directory if not exists
    if not os.path.exists(hp.trained_model_path):
        os.mkdir(hp.trained_model_path)

    path = os.path.join(hp.trained_model_path, './model.pth')
    torch.save(model.state_dict, path)



main()
