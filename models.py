import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import math
from config import cfg


# Convolutional Neural Network + MLP for FOG Detection
class CNN_FOG(nn.Module):
    def __init__(self, in_channels, seq_len, dropout_rate=cfg['DROPOUT'], n_heads=cfg['N_HEADS'],
                 n_enc_layers=cfg['N_ENC_LAYERS'], max_conv_filters=cfg['N_MAX_CONV_FILTERS']):
        '''
        Convolutional Transformer model
        :param in_channels: Int, number of input channels from features
        :param seq_len: Int, length of input sequence
        :param dropout_rate: Float, probability of neuron dropout for Dropout layers
        '''
        super(CNN_FOG, self).__init__()
        # TODO: Either sequentially pass time step input or increase first dim size?
        self.conv_block = ConvolutionalBlock(in_channels, max_conv_filters)

        # Transformer Block
        # self.transformer_enc = TransformerEncoder(seq_len=seq_len, embed_dim=32)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=max_conv_filters//4, nhead=n_heads)
        # self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        self.global_avg_pool = nn.AvgPool1d(kernel_size=max_conv_filters//4)

        # MLP Head TODO: Maybe set different dropout rate here than rest of network
        self.mlp = nn.Sequential(nn.Linear(seq_len, 80),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(80, 40),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(40, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        # reshaping to apply 1D conv for each timestep using 2D conv
        x = torch.swapaxes(x, -3, -1)
        conv_out = self.conv_block(x)
        conv_out = torch.swapaxes(conv_out, -3, -1)
        conv_out = conv_out.squeeze(dim=2)
        # transformer_out = self.transformer_enc(conv_out)
        gap_out = self.global_avg_pool(conv_out)
        gap_out = gap_out.squeeze(dim=-1)
        mlp_out = self.mlp(gap_out)
        mlp_out = mlp_out.squeeze()

        return mlp_out


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, max_conv_filters):
        '''
        Convolutional block containing 1D convolutional layers to extract embeddings
        :param in_channels: Int, number of input channels from features
        :param seq_len: Int, length of input sequence
        '''
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=max_conv_filters,
                               kernel_size=(4, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=max_conv_filters, out_channels=max_conv_filters//2,
                               kernel_size=(4, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=max_conv_filters//2, out_channels=max_conv_filters//4,
                               kernel_size=(4, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, cfg['N_WINDOWS']))  # Global avg pooling

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim, num_layers=3, n_heads=3, dropout_rate=cfg['DROPOUT'], expansion_factor=4):
        '''
        Transformer encoder block as presented in Vaswani et. al (2017)
        :param seq_len: Int, length of input sequence
        :param embed_dim: Int, number of dimensions of input embedding
        :param num_layers: Int, number of sequential EncoderLayer objects
        :param n_heads: Int, number of parallel attention heads
        :param dropout_rate: Float, probability of neuron dropout for Dropout layers
        :param expansion_factor: Int, factor to determine inner dimension for feed-forward network
        '''
        super(TransformerEncoder, self).__init__()

        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([EncoderLayer(embed_dim=embed_dim, n_heads=n_heads,
                                                  dropout_rate=dropout_rate, expansion_factor=expansion_factor)
                                     for _ in range(num_layers)])

    def forward(self, x):
        # out = self.positional_encoder(x)
        out = x
        for layer in self.layers:
            out = layer(out, out, out)  # self-attention only

        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=3, dropout_rate=cfg['DROPOUT'], expansion_factor=4):
        '''
        Encoder layer using as Transformer Encoder.
        :param embed_dim: Int, Number of dimensions of input embedding
        :param n_heads: Int, number of parallel attention heads
        :param dropout_rate: Float, probability of neuron dropout for Dropout layers
        :param expansion_factor: Int, factor to determine inner dimension for feed-forward network
        '''
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # TODO: Could change inner dim to 16 (const)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, expansion_factor * embed_dim),
                                          nn.ReLU(),
                                          nn.Linear(expansion_factor * embed_dim, embed_dim))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, k, q, v):
        attention_out = self.attention(k, q, v)
        attention_res_out = attention_out + v
        norm1_out = self.norm1(attention_res_out)
        dropout1_out = self.dropout1(norm1_out)

        feed_forward_out = self.feed_forward(dropout1_out)
        feed_forward_res_out = feed_forward_out + dropout1_out
        norm2_out = self.norm2(feed_forward_res_out)
        dropout2_out = self.dropout2(norm2_out)

        return dropout2_out


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=3):
        '''
        Multi-head attention layer for use in Transformer models.
        :param embed_dim: Int, number of dimensions in input embedding
        :param n_heads: Int, number of parallel heads
        '''
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key_vec, query_vec, value_vec):
        batch_size = key_vec.size(0)
        seq_len = key_vec.size(1)

        key_vec = key_vec.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
        query_vec = query_vec.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
        value_vec = value_vec.view(batch_size, seq_len, self.n_heads, self.single_head_dim)

        k = self.k(key_vec)
        q = self.q(query_vec)
        v = self.v(value_vec)

        # (batch_size, n_heads, seq_len, single_head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        k_adj = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_len)

        product = torch.matmul(q, k_adj)
        # Retain unit variance
        product = product / math.sqrt(self.single_head_dim)

        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, self.single_head_dim * self.n_heads)

        output = self.out(concat)

        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        '''
        Position embeddings for use in Attention layers.
        :param seq_len: Int, length of input sequence
        :param embed_dim: Int, number of dimensions in input embedding
        '''
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(seq_len, self.embed_dim)
        for pos in range(seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * 1) / self.embed_dim))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))

        pe = pe.unsqueeze(0)
        # Add to state_dict and register as buffer to prevent training updates
        # Remove if positional embeddings need to be learnable
        self.register_buffer('pe', pe)

    def forward(self, x, requires_grad=False):
        # Make input embedding larger than pos embeddings
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=requires_grad)

        return x
