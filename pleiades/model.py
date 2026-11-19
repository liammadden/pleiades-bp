"""
Simple Transformer model with:
 - Token embedding
 - Positional embedding
 - Causal/masked/autoregressive self-attention heads
 - Multilayer (two-layer) perceptrons with GELU activation
 - Residual/skip connections
 - Post-layer normalization (after each residual connection)
 - Context unembedding

Given an input sequence of n token indices, the model outputs
a logit vector of length equal to the vocabulary size. Given
an input batch of size B x T, where T is the sequence length,
the model outputs a logit tensor of size B x T x V where V is
the vocabulary size. Thus, the model is given batch[:, :-1],
and the inputs to cross entropy are
    output.contiguous().view(-1, vocab_size)
and
    batch[:, 1:].contiguous().view(-1)
for next token prediction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, seq_length, vocab_size, n_layer, n_head, embed_dim, head_dim, mlp_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, seq_length)
        self.layers = nn.ModuleList(
            [Block(embed_dim, head_dim, n_head, mlp_dim) for i in range(n_layer)],
            )
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encode(x)
        for i, block in enumerate(self.layers):
            x = block(x)
        x = self.unembed(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=5000):
        super().__init__()
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, head_dim, n_head, mlp_dim):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, head_dim, n_head)
        self.mlp = MLP(head_dim * n_head, mlp_dim, embed_dim)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = F.layer_norm(x, (x.size(-1),))
        x = x + self.mlp(x)
        x = F.layer_norm(x, (x.size(-1),))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, n_head):
        super().__init__()
        self.head_dim = head_dim
        self.n_head = n_head
        self.W_q = nn.Linear(embed_dim, head_dim * n_head, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim * n_head, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim * n_head, bias=False)
    
    # this can be done more efficiently with F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # but it doesn't support forward AD yet
    def scaled_dot_product_attention(self, query, key, value):
        T = query.size(-2)
        attn_bias = torch.zeros(T, T, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(T, T, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1)) + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value)

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embed_dim
        head_dim = self.head_dim
        n_head = self.n_head
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # for scaled_dot_product_attention we need the second to last
        # axis to index position within a sequence, and the last axis
        # to index elements of a head's embedding.
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        x = self.scaled_dot_product_attention(q, k, v)
        # x is B x n_head x T x head_dim. We want it to be B x T x head_dim * n_head.
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, inner_dim)
        self.output = nn.Linear(inner_dim, output_dim)
    
    def forward(self, x):
        x = F.gelu(self.hidden(x))
        x = self.output(x)
        return x