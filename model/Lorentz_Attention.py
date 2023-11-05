import math
import torch
import torch.nn as nn
from manifolds.lorentz import Lorentz


class LorentzSelfAttention(nn.Module):

    def __init__(self, dimension, dropout=0):
        super(LorentzSelfAttention, self).__init__()
        self.d_emb = dimension
        self.manifold = Lorentz()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dimension)]))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, Q, K=None, V=None, mask=None):
        if K is not None and V is not None:
            query, key, value = Q, K, V
        else:
            query, key, value = Q, Q, Q
        batch_size = key.size(0)

        def shape(x):
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, 1, self.d_emb)
            return x.transpose(1, 2)

        key = shape(key)
        value = shape(value)
        query = shape(query)
        key_len = key.size(2)
        inf = -2 ** 32 + 1
        attn = (2 + 2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, key_len)
            tri_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(pad_mask.device)
            mask = tri_mask + pad_mask
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, inf)

        attn = self.softmax(attn)
        latent_emb = self.manifold.mid_point(value, attn)
        latent_emb = latent_emb.transpose(1, 2).squeeze(2)
        output = latent_emb

        return output
