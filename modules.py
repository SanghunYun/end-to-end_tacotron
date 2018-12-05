# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import torch
from torch.autograd import Variable
from torch import nn

dtype = torch.float
vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''
vocab_size=len(vocab)
har_to_ix = {char: i for i, char in enumerate(vocab)}

class embed(nn.Module):

    """
    input text to character embedding
    input : A tensor with 1 or more dimension ((batch size), Text length)
    num_units : length of the text input which is padded (# of hidden units) - 188
    embed_size : embed 1 character to 256 embed size - 256
    """

    def __init__(self, vocab_size, num_units, zero_pad=True):
        super(embed,self).__init__()
        self.lookup_table=nn.Embedding(vocab_size, embed_size)
        self.vocab_size=vocab_size
        self.num_units=num_units
        self.zero_pad=zero_pad

    def forward(self,text):
        #batch size떄문에 3차원으로 다시 만들어야 함

        #initialize the rst
        rst = torch.empty(hp.Tx,hp.embed_size, dtype=torch.float32)
        
        #find embedding in lookup_table
        for idx,i in enumerate(text):
            rst[idx]=self.lookup_table(torch.tensor(vocab.index(i), dtype=torch.long))
        
        ##if zero_pad==True, add zeros to first row
        if self.zero_pad:
            rst=torch.cat((torch.zeros(1,self.num_units),rst[1:,:]),0)
        return rst

class bn(nn.Module):

    """
    batch normalization on the (N,Tx)


    """

    def __init__(self, activation_fn=None):
        super(bn,self).__init__()      

    def forward(self, input):
        n_features=input.shape[2]   #E/2    ->  batch norm should be done except last dim (E/2) (input:(N,Tx,E/2))  
        input_t=torch.transpose(input,1,2)  #E/2(2) to Position of C(1)
        batch_norm=nn.BatchNorm1d(n_features)
        return torch.transpose(batch_norm(input_t),2,1) #restore shape

class conv1d(nn.Modlue):

    """
    input : (N,Tx, E/2)
    """

    def __init__(self, filters=None,size=1,rate=1,padding="SAME",use_bias=False,activation_fn=None):
        super(bn,self).__init__()
        self.size=size
        self.rate=rate
        self.padding=padding
        self.use_bias=use_bias
        self.activation_fn=activation_fn      

    def forward(self, input):
        #padding=causal은 아직 없는 것 같아서 일단 안만듦
        #filter=None도 일단 생략
        #(N,C,L) - N,channel(E/2), length(Tx)
        input_t=torch.transpose(input,1,2)
        if self.padding == "SAME":
            pad_size = max(int((size-1)/2),int(size/2))
        else:
            pad_size=0
        if self.activation_fn != None:
            return torch.transpose(torch.nn.conv1d(input_t,kernel_size=self.size, stride=self.rate, padding=pad_size bias=self.use_bias),2,1)
        else:
            #activation function 아직 구현 안함
            return torch.transpose(torch.nn.conv1d(input_t,kernel_size=self.size, stride=self.rate, padding=pad_size, bias=self.use_bias),2,1)
    



        
        


            


    
    


    