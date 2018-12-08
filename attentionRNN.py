import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import hyperparams as hp

class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism (Vinyals et al.)
    """
    def __init__(self, num_units):
        """
        :param num_units: dimension of hidden units
        """
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units

        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias=False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)

        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)

        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, hp.num_mels * hp.outputs_per_step)

    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden, gru2_hidden):

        memory_len = memory.size()[1]
        batch_size = memory.size()[0]

        # Get keys
        keys = self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)

        # Get hidden state (query) passed through GRUcell
        d_t = self.attn_grucell(decoder_input, attn_hidden)

        # Duplicate query with same dimension of keys for matrix operation (Speed up)
        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)

        # Calculate attention score and get attention weights
        attn_weights = self.v(F.tanh(keys + d_t_duplicate).view(-1, self.num_units)).view(-1, memory_len, 1)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)

        # Concatenate with original query
        d_t_prime = torch.bmm(attn_weights.view([batch_size,1,-1]), memory).squeeze(1)

        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1))
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        gru2_input = gru1_input + gru1_hidden

        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input + gru2_hidden

        # Output
        output = self.out(bf_out).view(-1, hp.num_mels, hp.outputs_per_step)

        return output, d_t, gru1_hidden, gru2_hidden

    def inithidden(self, batch_size):
        if use_cuda:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
        else:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)

        return attn_hidden, gru1_hidden, gru2_hidden