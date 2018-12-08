from hyperparams import Hyperparams as hp
import torch
import torch.nn as nn
from modules import *
from attentionRNN import *

# import modules import *

class transcript_encoder(nn.Module):
    def __init__(self, shape, is_training=True):
        super(transcript_encoder, self).__init__()
        self.shape=shape
        self.prenet = prenet(self.shape) #nn.Module에서 알아서 prop... is_training 필요 없을 듯
        self.shape=(self.shape[0],self.shape[1],self.shape[2]/2) #(N, Tx, E/2)

        self.conv1d_banks = conv1d_banks(K=hp.encoder_num_banks)
        self.shape=(self.shape[0],self.shape[1],self.shape[2]*hp.encoder_num_banks) #(N, Tx, E/2*K)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)          # pytorch
        self.conv1d_1 = conv1d(self.shape, filters=hp.embed_size//2, size=3)
        self.shape = (self.shape[0], self.shape[1], self.shape[2]//hp.encoder_num_banks) #(N, Tx, E/2)
        self.bn_1 = bn(self.shape[2], activation_fn="ReLU")

        self.conv1d_2 = conv1d(self.shape, filters=hp.embed_size//2, size=3) 
        #self.relu = nn.ReLU()
        #self.relu_1 = nn.ReLU()
        #self.relu_2 = nn.ReLU()
        
        self.bn_2 = bn(self.shape[2], activation_fn="ReLU")
        self.highwaynet = highwaynet(self.shape, num_units=hp.embed_size//2)
        self.gru = gru(channel = self.shape[2],time = self.shape[1], num_units=hp.embed_size//2, bidirection=True) #output : (N,Tx,E) - bidirectional 때문               # pytorch 
        
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
    input : (N, Ty, n_mels, 1)
     - (N,W,H,C) 꼴

    pytorch에선 (N,C,H,W) 꼴로 받아야 함 -> 그래서 transpose


    """

    def __init__(self, shape):
        super(reference_encoder, self).__init__()
        self.shape = shape # (N,Ty,n_mels,1)
        # half padding    
        # int(k-1/2)
        pad = int((3-1)/2)

        # 6-Layer Strided Conv2D -> (N, 128, n_mels/64, T/64)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=pad)
        #self.relu_1 = nn.ReLU()
        self.bn_1 = bn(1, activation_fn="ReLU")

        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=pad)
        #self.relu_2 = nn.ReLU()
        self.bn_2 = bn(32,activation_fn="ReLU")

        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=pad)
        #self.relu_3 = nn.ReLU()
        self.bn_3 = bn(64,activation_fn="ReLU")

        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=pad)
        #self.relu_4 = nn.ReLU()
        self.bn_4 = bn(64,activation_fn="ReLU")

        self.conv2d_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=pad)
        #self.relu_5 = nn.ReLU()
        self.bn_5 = bn(64,activation_fn="ReLU")

        self.conv2d_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=pad)
        #self.relu_6 = nn.ReLU()
        #(N, 128, n_mels/64, T/64)
        self.bn_6 = bn(128,activation_fn="ReLU")

        self.gru = gru(channel = 128, time = self.shape[1]/64, num_units=128, bidirection=False)

        # tf.layers.dense(inputs, units(output))
        self.dense = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
    
    def forward(self, inputs):

        input_t = torch.Tensor.transpose(inputs,1,3) # transpose to (N,C,H,W)
        # conv2d X 6  (N, 1, Ty, n_mels)  -->  (N, 128, Ty/64, n_mels/64, 1)

        tensor = self.conv2d_1(input_t)
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
        tensor = torch.Tensor.cat(channels, dim=2)

        # GRU (N, T/64, 128*n_mels/64)  -->  (N, 128)
        tensor = self.gru(tensor)
        tensor = tensor[:, -1, :]

        # FC --> (N, 128)
        prosody = self.dense(tensor)
        prosody = self.tanh(prosody)

        return prosody



class decoder1(nn.Module):
    
    """
        inputs : (N, Ty/r, n_mels)
        memory : (N, Tx, E)
        return : (N, Ty/r, n_mels*r)

        Referenced https://github.com/r9y9/tacotron_pytorch
    """
    
    def __init__(self, shape, is_training): #여기서 shape는 Go의 shape여야 함
        super(decoder1, self).__init__()
        self.is_training = is_training
        self.shape = shape
        self.prenet = prenet(self.shape)
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttention(256)
        )
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.proj_to_mel = nn.Linear(256, hp.n_mels*hp.r)

    def forward(self, inputs, memory):
        N = memory.size(0)
        processed_memory = self.memory_layer(memory)
        memory_lengths = len(memory)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None
        
        # Run greedy decoding if inputs is None (at Training time)
        greedy = (self.is_training == True)

        if self.is_training==False:
            # Grouping multiple frames if necessary
            if inputs.size(-1) == hp.n_mels:
                inputs = inputs.view(N, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == hp.n_mels * self.r
            T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(
            memory.data.new(N, hp.n_mels).zero_())

         # Init decoder states
        attention_rnn_hidden = Variable(
            memory.data.new(N, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            memory.data.new(N, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            memory.data.new(N, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        if self.is_training==False:
            inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input

        while True:
            if t > 0:
                current_input = outputs[-1,:,-hp.n_mels] if greedy else inputs[t - 1,:,-hp.n_mels]
            # Prenet
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                memory, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.Tensor.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = decoder_input
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1

            if t >= T_decoder:
                break
        assert greedy or len(outputs) == T_decoder

        alignments = torch.Tensor.stack(alignments).transpose(0, 1) #(Stack(Ty/r),N,E)->(N,Ty/r,E)
        outputs = torch.Tensor.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments

        """
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
        """

class decoder2(nn.Module):

    """
        inputs : (N, Ty/r, n_mels*r)
        output : (N, Ty, 1+n_fft//2)
    """

    def __init__(self,shape):
        super(decoder2, self).__init__()
        self.shape = shape
        self.conv1d_banks = conv1d_banks(K=hp.decoder_num_banks) # (N, Ty, E*K/2)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1)          # pytorch

        self.shape(shape[0],shape[1],hp.embed_size*hp.decoder_num_banks//2)

        self.conv1d_1 = conv1d(shape, filters=hp.embed_size // 2, size=3) # (N, Ty, E/2)
        self.shape = (shape[0],shape[1], hp.embed_size//2) # (N, Ty, E/2)
        self.bn_1 = bn(shape[2], activation_fn="ReLU")

        self.conv1d_2 = conv1d(shape, filters=hp.n_mels, size=3)
        self.bn_2 = bn(shape[2])

        self.dense = nn.Linear(hp.n_mels,hp.embed_size//2)
        self.highwaynet = highwaynet(shape, num_units=hp.embed_size//2)
        self.gru = gru(channel = shape[2],time=shape[1], bidirection=True)

    def forward(self, inputs):
        # (N, Ty/r,  n_mels*r)  -->  (N, Ty, n_mels)
        inputs = inputs.view([*inputs.shape][0], -1, hp.n_mels)

        # conv1d bank
        dec=self.conv1d_banks(K=hp.decoder_num_banks)

        # Max pooling
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