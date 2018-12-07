from hyperparams import Hyperparams as hp
import torch
import torch.nn as nn

# import modules import *

class transcript_encoder(nn.Module):
    def __init__(self, is_training=True):
        super(transcript_encoder, self).__init__()

        self.is_training = is_training

        self.prenet = prenet(is_training=self.is_training)
        self.conv1d_banks = conv1d_banks(K=hp.encoder_num_banks, is_training=self.is_training)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)          # pytorch
        
        self.conv1d_1 = conv1d(filters=hp.embed_size//2, size=3)  
        self.conv1d_2 = conv1d(filters=hp.embed_size//2, size=3) 
        self.relu = nn.ReLU()
        #self.relu_1 = nn.ReLU()
        #self.relu_2 = nn.ReLU()
        self.bn_1 = bn(is_training=is_training, activation_fn=self.relu)
        self.bn_2 = bn(is_training=is_training, activation_fn=self.relu)

        self.gru = gru(num_units=hp.embd_size//2, bidirection=True)                     # pytorch
        self.highwaynet = highwaynet(num_units=hp.embed_size//2)
    def forward(self, inputs):
        # prenet
        prenet_out = self.prenet(inputs)

        # CBHG
        # Conv1d banks
        enc = self.conv1d_banks(prenet_out)

        # Max pooling
        enc = self.max_pool(enc)

        ## Conv1d projections
        enc = self.conv1d_1(enc)
        enc = self.bn_1(enc)
        
        enc = self.conv1d_2(enc)
        enc = self.bn_2(enc)

        enc += prenet_out

        ## Highway
        for i in range(hp.num_highwaynet_blocks):
            enc = self.highwaynet(enc)

        ## Bidirectional GRU
        texts = self.gru(enc)

        return texts



class reference_encoder(nn.Module):

    """
        기존 tensorflow
        inputs : (N, Ty, n_mels, 1)
                batch, lenght, n_mels, channel
                NHWC

        pytorch
        inptus : (N, 1, Ty, n_mels)
                batch, channel, lenght, n_mels

        train.py Line 44
        기존 : N H W C
        바꿈 : N C H W
        .unsqueeze(1)  -->  c 확장 위치


    """

    def __init__(self, is_training):
        super(reference_encoder, self).__init__()
        self.is_training = is_training
        
        # half padding    int(k-1/2)
        pad = int((3-1)/2)        

        self.relu = nn.ReLU

        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kerenl_size=3, stride=2, padding=pad)
        #self.relu_1 = nn.ReLU()
        self.bn_1 = bn(is_training=is_training, activation_fn=self.relu)

        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=pad)
        #self.relu_2 = nn.ReLU()
        self.bn_2 = bn(is_training=is_training, activation_fn=self.relu)

        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=pad)
        #self.relu_3 = nn.ReLU()
        self.bn_3 = bn(is_training=is_training, activation_fn=self.relu)

        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=pad)
        #self.relu_4 = nn.ReLU()
        self.bn_4 = bn(is_training=is_training, activation_fn=self.relu)

        self.conv2d_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=pad)
        #self.relu_5 = nn.ReLU()
        self.bn_5 = bn(is_training=is_training, activation_fn=self.relu)

        self.conv2d_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=pad)
        #self.relu_6 = nn.ReLU()
        self.bn_6 = bn(is_training=is_training, activation_fn=self.relu)

        self.gru = gru(num_units=128, bidirection=False)

        # tf.layers.dense(inputs, units(output))
        self.dense = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
    
    def forward(self, inputs):
        # conv2d X 6  (N, 1, Ty, n_mels)  -->  (N, 128, Ty/64, n_mels/64)
        tensor = self.conv2d_1(inputs)
        tensor = self.bn_1(tensor)

        tensor = self.conv2d_2(tensor)
        tensor = self.bn_2(tensor)

        tensor = self.conv2d_3(tensor)
        tensor = self.bn_3(tensor)

        tensor = self.conv2d_4(tensor)
        tensor = self.bn_4(tensor)

        tensor = self.conv2d_5(tensor)
        tensor = self.bn_5(tensor)

        tensor = self.conv2d_6(tensor)
        tensor = self.bn_6(tensor)

        # Unroll (N, 128, Ty/64, n_mels/64)  -->  (N, T/64, 128*n_mels/64)
        N, C, H, W = [*tensor.shape]
        channels = []
        for i in range(0, C):
            channels.append(tensor[:, i, :, :])
        tensor = torch.cat(channels, dim=2)

        # GRU (N, T/64, 128*n_mels/64)  -->  (N, 128)
        tensor = self,gru(tensor)
        tensor = tensor[:, -1, :]

        # FC --> (N, 128)
        prosody = self.dense(tensor)
        prosody = self.tanh(prosody)

        return prosody



class decoder1(nn.Module):
    
    """
        inputs : (N, Ty/r, n_mels*r)
        memory : (N, Tx, E)

        return : (N, Ty/r, n_mels*r)
    """
    
    def __init__(self, is_training):
        super(decoder1, self).__init__()

        self.is_training = is_training
        
        self.prenet = prenet(self.is_training)
        self.attention_decoder(num_units=hp.embed_size)
        self.gru_1 = gru(num_units=hp.embed_size, bidirection=False)
        self.gru_2 = gru(num_untis=hp.embed_size, bidirection=False)
        self.dense = nn.Linear(hp.embed_size, hp.n_mels*hp.r)

    def forward(self, inputs):
        # Decoder prenet  (N, Ty/r, n_mels*r)  -->  (N, Ty/r, E/2)
        inputs = self.prenet(inputs) 

        # Attention RNN   (N, Ty/r, E/2), (N, Tx, E+128)  -->  (N, Ty/r, E)
        dec, state = attention_decoder(inputs, memory)
        
        ## for attention monitoring
        #TODO:  tf.transpose 했던거 pytorch처럼 전치시켜줘야함
        # alighnmets = (state.alignment_history.stack(), [1,2,0])

        dec += self.gru_1(dec)
        dec += self.gru_2(dec)

        # Outputs => (N, Ty/r, n_mels*r)
        mel_hats = self.dense(dec)

        return mel_hats #, alignments

class decoder2(nn.Module):

    """
        inputs : (N, Ty/r, n_mels*r)
        output : (N, Ty, 1+n_fft//2)
    """

    def __init__(self, is_training):
        super(decoder2, self).__init__()
        self.is_training = is_training
        
        self.conv1d_banks = conv1d_banks(K=hp.decoder_num_banks, self.is_training)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1)          # pytorch

        self.relu = nn.ReLU()
        self.conv1d_1 = conv1d(filters=hp.embed_size // 2, size=3)
        self.bn_1 = bn(is_training=self.is_training, activation_fn=self.relu)

        self.conv1d_2 = conv1d(filters=hp.n_mels, size=3)
        self.bn_2 = bn(is_training=self.is_training)

        self.dense = nn.Linear(hp.n_mels,hp.embed_size//2)
        self.highwaynet = highwaynet(num_units=hp.embed_size//2)
        self.gru = gru(hp.embed_size//2, bidirection=True)


    def forward(self, inputs):
        # (N, Ty/r,  n_mels*r)  -->  (N, Ty, n_mels)
        inputs = inputs.view([*inputs.shape][0], -1, hp.n_mels)

        # conv1d bank
        dec = self.max_pool1d(dec)

        ## conv1d projections
        dec = self.conv1d_1(dec)
        dec = self.bn_1(dec)

        dec = self.conv1d_2(dec)
        dec = self.bn_2(dec)

        # FC
        dec = self.dense(dec)

        # Highwaynet
        for i in range(4):
            dec = self.highwaynet(dec)

        # Bidirectional GRU
        dec = self.gru(dec)

        # ouputs  (N, Ty, 1+n_fft//2)
        outputs = self.dense(dec)

        return outputs