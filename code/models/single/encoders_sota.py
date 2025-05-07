"""
Lightweight Temporal Attention Encoder module
> source: https://github.com/VSainteuf/lightweight-temporal-attention-pytorch
> paper:  https://doi.org/10.1007/978-3-030-65742-0_12

Temporal Attention Encoder module
> source: https://github.com/VSainteuf/pytorch-psetae
> paper: https://doi.org/10.1109/CVPR42600.2020.01234

TempCNN - Pytorch re-implementation of Pelletier et al. 2019
> source: https://github.com/charlotte-pel/temporalCNN
> paper: https://doi.org/10.3390/rs11050523

ConvTran - Pytorch implementation of ConvTran
> source: https://github.com/Navidfoumani/ConvTran/
> paper: https://link.springer.com/content/pdf/10.1007/s10618-023-00948-2.pdf
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from .base_encoders import Base_Encoder
from .sota_aux.tae import MultiHeadAttention, MultiHeadAttention_MQ, get_sinusoid_encoding_table
from .sota_aux.tcnn import Conv1D_BatchNorm_Relu_Dropout, FC_BatchNorm_Relu_Dropout
from .sota_aux.convtran import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding, Attention, Attention_Rel_Scl, Attention_Rel_Vec

class LTAE(Base_Encoder):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, mask_nans: bool=False, **kwargs):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att
        self.mask_nans = mask_nans

        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention_MQ(
            n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3)*x.size(4))

        sz_b, seq_len, d = x.shape

        if self.mask_nans: 
             masks = torch.stack( [ (torch.isfinite( v ).sum(axis=-1) ==0) for v in x], axis=0 ) 
             x = torch.nan_to_num(x, nan=0.0) 
        else:
            masks = []

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        enc_output = x + self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output, masks=masks)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return {"rep": enc_output, "attention" : attn}
        else:
            return {"rep": enc_output}

    def get_output_size(self):
        return self.n_neurons[-1]


class TemporalAttentionEncoder(Base_Encoder):
    def __init__(self, in_channels=128, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24, positions=None, return_att=False, mask_nans=False, **kwargs):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model

        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att
        self.name = 'TAE_dk{}_{}Heads_{}_T{}_do{}'.format(d_k, n_head, '|'.join(list(map(str, self.n_neurons))), T,
                                                          dropout)
        self.mask_nans = mask_nans

        if positions is None:
            positions = len_max_seq + 1
        else:
            self.name += '_bespokePos'

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(positions, self.in_channels, T=T),
            freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
            self.name += '_dmodel{}'.format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == n_head * self.d_model)
        #assert (self.n_neurons[-1] == self.d_model)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           nn.ReLU()])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3)*x.size(4))

        sz_b, seq_len, d = x.shape

        if self.mask_nans: 
             masks = torch.stack( [ (torch.isfinite( v ).sum(axis=-1) ==0) for v in x], axis=0 ) 
             x = torch.nan_to_num(x, nan=0.0) 
        else:
            masks = []

        x = self.inlayernorm(x)
        
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        enc_output = x + self.position_enc(src_pos)

        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output, masks=masks)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return {"rep": enc_output, "attention" : attn}
        else:
            return {"rep": enc_output}

    def get_output_size(self):
        return self.n_neurons[-1]


class TempCNN(Base_Encoder):
    def __init__(self, input_dim, sequence_length, kernel_size=5, hidden_dims=64, dropout=0.5, n_layers=3, **kwargs):
        super(TempCNN, self).__init__()

        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length

        layers = []
        for i in range(n_layers):
            if i ==0:
                inp_dim_i = input_dim
            else:
                inp_dim_i = self.hidden_dims
            layers.append(Conv1D_BatchNorm_Relu_Dropout(inp_dim_i, self.hidden_dims, kernel_size=kernel_size, drop_probability=dropout))
        self.conv_layers = nn.Sequential(*layers)
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims*sequence_length, 4*self.hidden_dims, drop_probability=dropout)

    def forward(self,x):
        if len(x.shape) == 5:
            x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3)*x.size(4))
        
        x = x.permute(0,2,1)
        x = self.conv_layers(x)
        x_flatten = x.view(x.size(0), -1)
        return self.dense(x_flatten)

    def get_output_size(self):
        return 4*self.hidden_dims


class ConvTran(nn.Module):
    #in which order forward?
    def __init__(self, 
                 feature_size: int,
                 layer_size: int = 128,
                 num_layers: int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 len_max_seq: int = 24,
                 pos_encoding: str = 'tAPE',
                 rel_pos_encoding: str = 'eRPE',
                 dim_feedforward = 512,
                 **kwargs):
        super().__init__()
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.Fix_pos_encode = pos_encoding
        self.Rel_pos_encode = rel_pos_encoding
        # Embedding Layer ----------------------------------------------------------- (transformer)
        self.embed_layer = nn.Sequential(nn.Conv2d(1, self.layer_size*4, kernel_size=[np.minimum(8,len_max_seq), 1], padding='same'),
                                         nn.BatchNorm2d(self.layer_size*4),
                                         nn.GELU())
        out_dim = self.layer_size*4
        for i in range(np.maximum(0, num_layers-2)):
            in_dim = self.layer_size*np.maximum((4-i),1)
            out_dim = self.layer_size*np.maximum((3-i),1)
            self.embed_layer = nn.Sequential(*list(self.embed_layer.children()), 
                                             nn.Conv2d(in_dim, out_dim, kernel_size=[np.minimum(8,len_max_seq), 1], padding='same'),
                                            nn.BatchNorm2d(out_dim),
                                            nn.GELU())
            

        self.embed_layer2 = nn.Sequential(nn.Conv2d(out_dim, self.layer_size, kernel_size=[1, feature_size], padding='valid'),
                                          nn.BatchNorm2d(self.layer_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(self.layer_size, dropout=dropout, max_len=len_max_seq)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(self.layer_size, dropout=dropout, max_len=len_max_seq)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(self.layer_size, dropout=dropout, max_len=len_max_seq)
        else:
            self.Fix_Position = nn.Identity()

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(self.layer_size, num_heads, len_max_seq, dropout)
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(self.layer_size, num_heads, len_max_seq, dropout)
        else:
            self.attention_layer = Attention(self.layer_size, num_heads, dropout)

        self.LayerNorm = nn.LayerNorm(self.layer_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(self.layer_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(self.layer_size, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.layer_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3)*x.size(4))

        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(3)

        x_src = x_src.permute(0, 2, 1)

        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)

        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)

        return {"rep": self.flatten(out)}

    def get_output_size(self):
        return self.layer_size 
    

