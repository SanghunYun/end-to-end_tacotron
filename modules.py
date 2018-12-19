# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from torch import *
from torch.autograd import Variable
from torch import nn
import attentionRNN
import torch
import numpy as np

vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''
vocab_size=len(vocab)
har_to_ix = {char: i for i, char in enumerate(vocab)}
use_cuda = torch.cuda.is_available()

class embed(nn.Module):

    """
    input text to character embedding
    inputs : A tensor with 1 or more dimension ((batch size), Text length)
    already indexed at the data_load()
    
    num_units : length of the text input which is padded (# of hidden units) - 188
    
    embed_size : embed 1 character to 256 embed size - 256

    output : (N, Tx, E)
    """

    def __init__(self, vocab_size, num_units, zero_pad=True):
        super(embed,self).__init__()
        self.lookup_table = nn.Embedding(vocab_size, num_units)
        self.zero_pad=zero_pad

    def forward(self,inputs):
        
        ##if zero_pad==True, add zeros to first row
        ## - 일단 없어서 구현 안함
        if use_cuda:
            for i in range(inputs.shape[0]):
                print(inputs[i])
            try:
                a = self.lookup_table(torch.tensor(inputs, dtype=torch.long).cuda())
            except ZeroDivisionError as ex:
                print(ex)
            print('============================')
            print(a)
            return self.lookup_table(torch.tensor(inputs, dtype=torch.long).cuda())
        else:
            return self.lookup_table(torch.tensor(inputs, dtype=torch.long))

class bn(nn.Module):

    """

    if inputs = 3D
    batch norm should be done except last dim (E/2) (input:(N,Tx,E/2)) 

    batch normalization on the (N,Tx)
    So, num_features(C of N,C,L) at nn.BatchNorm1d : input.shape[2] (E/2)

    Transpose E/2(2) to Position of C(1)
    (pytorch에서는 (N,C,L)이 input 받는 형태)

    """

    def __init__(self, channel, activation_fn=None):
        super(bn,self).__init__() 
        self.activation_fn = activation_fn
        if self.activation_fn == "ReLU":
            self.relu = torch.nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(channel) # networks.py의 ref encoder의 경우
        self.batch_norm2d = nn.BatchNorm2d(channel)
    def forward(self, inputs):
        if inputs.dim()==3:
            input_t=torch.transpose(inputs,1,2) 

            #restore shape
            rst = torch.transpose(self.batch_norm(input_t),2,1)
        elif inputs.dim()==4:
            rst = self.batch_norm2d(inputs)
            
        if self.activation_fn == "ReLU":
            return self.relu(rst)
        else:
            return rst  

class conv1d(nn.Module):

    """
    inputs : (N,Tx, E/2)
    channel : shape[2]
        
    (N,C,L) - N,channel(E/2), length(Tx)로 바꾼 후 conv계산 -> 다시 N,Tx,E/2 꼴로 바꿔줘야 함
    pytorch에서는 in_channel이 shape[1], tensorflow에서는 in_channel이 shape[2], shape[0]이 batch 크기인 건 공통
       
    https://discuss.pytorch.org/t/output-shape-of-conv1d-in-pytorch-and-keras-are-different/3398 참고
    """

    def __init__(self, channel, filters=None, size=1, rate=1, padding="SAME", use_bias=False, activation_fn=None):
        super(conv1d,self).__init__()
        self.activation_fn=activation_fn
        self.relu = torch.nn.ReLU()

        #filters=None이면 filter의 수는 input channel의 수와 같게 (E/2)
        if filters == None:
            filters = channel


        if padding == "SAME":
            pad_size = int((size-1) / 2)
        else:
            pad_size=0

        self.conv = torch.nn.Conv1d(channel, filters, kernel_size=size, stride=rate, padding=pad_size, bias=use_bias)

    def forward(self, inputs):

        #padding=causal은 아직 안 쓰이는 것 같아서 일단 안만듦

        #channel size should be E/2, which is at shape[2]. So reshape it to locate at shape[1]
        rst = torch.transpose(self.conv(torch.transpose(inputs, 1, 2)),2,1)

        if self.activation_fn == "ReLU":
            return self.relu(rst)
        else:
            return rst
        
# 여기부터는 좀 확인 필요
class conv1d_banks(nn.Module):
    
    """
    inputs : (N,Tx,E/2)
    output : (N,Tx,K*E/2)
    K : size of Banks (filter size)...like N gram
    iteratively apply conv1d and apply relu

    rst : concat the conv1 results from size 1 to K
    """

    def __init__(self,K, position):
        super(conv1d_banks,self).__init__()
        self.conv_list = nn.ModuleList()
        self.K = K
        self.position = position
        if self.position == 'tr_encoder':
            for i in range(K): # 0~K-1 => size=i+1
                self.conv_list.append(conv1d(hp.embed_size // 2,hp.embed_size // 2,size=i+1,padding="SAME"))
        elif self.position == 'decoder2':

            #self.conv_list.append(conv1d(hp.n_mels, hp.embed_size//2, size=1, padding='SAME'))
            for i in range(0, K):
                self.conv_list.append(conv1d(hp.n_mels, hp.embed_size // 2, size=i+1, padding='SAME'))
        
        self.bn = bn((hp.embed_size//2)*K, activation_fn="ReLU")
        
    
        
    def forward(self,inputs):
        rst = self.conv_list[0](inputs)
        for i in range(1,self.K): # 1~K-1 -> size 2~K
            output = self.conv_list[i](inputs) # (N,Tx,E/2) 꼴로 나옴
            
            if self.position == 'tr_encoder':
                pad_size = (hp.batch_size, 1, 128)
            elif self.position == 'decoder2':
                pad_size = (hp.batch_size, 1, hp.embed_size//2)
            if i % 2 == 1 :
                zero_pad = np.zeros(pad_size, dtype=np.float32)
                zero_pad = torch.from_numpy(zero_pad).type(torch.cuda.FloatTensor)
                output = torch.cat((output, zero_pad), 1)
            rst = torch.cat((rst,output),2)
        
        rst = self.bn(rst)
        return rst

class gru(nn.Module):
    """
    applies gru

    inputs : (N,Time(T), Channel(E/2...))
/
    torch.nn.gru input : (T,N,E/2)

    num_units = # of hidden units passed (width of gru)

    num_layers = # of rnn layers (# of cells) - T
    
    input of torch.rnn.gru : (seq_len, batch, input_size) <- (T, N, channel)
    output of torch.rnn.gru : (seq_len, batch, num_directions(bidirectional이므로 2) * hidden_size)->(T, N, num_units)
    --> should be transposed (bidirectional의 경우 알아서 concat되어서 나오는 듯.., 확인 필요)

    output : (N,T,num_units) if bidirectional, num_units*2

    """


    def __init__(self, channel, num_units=None, bidirection=False):
        super(gru,self).__init__()
        if num_units == None:
            num_units = channel
        if bidirection == True:
            self.gru = torch.nn.GRU(input_size=channel, hidden_size=num_units, bidirectional=True)
        else:
            self.gru = torch.nn.GRU(input_size=channel, hidden_size=num_units, bidirectional=False)

    
    def forward(self,inputs):
        input_t = torch.transpose(inputs,0,1)
            #tensor_cat = torch.cat((tensor_cat, tensor[:, i, :, :]), 2)
        output,hn = self.gru(input_t)   #hn : hidden states for each t
        return torch.transpose(output,1,0)

class prenet(nn.Module):
    """

    inputs : (N, T(Tx or Ty/r), E/2)
    output : (N, T, num_units/2)

    """
    def __init__(self, in_dim, num_units=None):
        super(prenet,self).__init__()
        if num_units == None:
            num_units=[hp.embed_size,hp.embed_size//2]
        self.dense1 = torch.nn.Linear(in_dim,num_units[0])
        self.dropout1 = torch.nn.Dropout(p=hp.dropout_rate)
        self.dense2 = torch.nn.Linear(num_units[0],num_units[1])
        self.dropout2 = torch.nn.Dropout(p=hp.dropout_rate)

    def forward(self,inputs):
        rst = self.dense1(inputs)
        rst = self.dropout1(rst)
        rst = self.dense2(rst)
        rst = self.dropout2(rst)
        return rst


class highwaynet(nn.Module):
    """
    inputs : (N, T, W) - (ex> N, Tx or Ty/r, E/2)
    """
    def __init__(self, in_dim, num_units=None):
        super(highwaynet,self).__init__()
        if num_units == None:
            num_units = in_dim
        self.H = torch.nn.Linear(in_dim, num_units)
        self.T = torch.nn.Linear(in_dim,num_units)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,inputs):

        h_relu = self.relu(self.H(inputs))
        t_sigmoid = self.sigmoid(self.T(inputs))
        a = h_relu *t_sigmoid
        b = 1.-t_sigmoid
        c = inputs * b
        output = a + c
        #output = h_relu * t_sigmoid + inputs * (1.-t_sigmoid)

        return output
    




        
        


            


    
    


    