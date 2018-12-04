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
    def __init__(self, activation_fn=None):
        super(bn,self).__init__()      
    

    def forward(self, input):
        input_shape = input.size()
        input_rank = input.


            


    
    


    