import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention


class TanhAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)

    def reset_parameters(self):
        self.ws1.reset_parameters()
        self.ws2.reset_parameters()
        self.wst.reset_parameters()

    def forward(self, x, memory, memory_mask=None, fast_weights=None, **kwargs):
        if fast_weights is None:
            item1 = self.ws1(x)  # [nb, len1, d]
            item2 = self.ws2(memory)  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S_logit = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        else:
            item1 = F.linear(x, fast_weights['ws1.weight'], fast_weights['ws1.bias'])  # [nb, len1, d]
            item2 = F.linear(memory, fast_weights['ws2.weight'])  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S_logit = F.linear(torch.tanh(item), fast_weights['wst.weight']).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S_logit = S_logit.masked_fill(memory_mask == 0, -1e4)
        S = F.softmax(S_logit, -1)
        return torch.matmul(S, memory), S_logit  # [nb, len1, d], [nb, len1, len2]


class ContextQueryAttention(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.core_attention = TanhAttention(model_dim)
        self.ffn = nn.Linear(4 * model_dim, model_dim)

    def forward(self, context, query, query_mask=None):
        """
        :param context: (bs, max_context_len, model_dim)
        :param query: (bs, max_query_len, model_dim)
        :param query_mask: (bs, max_query_len)
        :return:
        """
        matrix_a, attn_logit = self.core_attention(context, query, query_mask)
        c_q_attn, q_c_attn = attn_logit.softmax(-1), attn_logit.softmax(1)
        matrix_b = c_q_attn.bmm(q_c_attn.transpose(-2, -1)).bmm(context)
        fusion = torch.cat((context, matrix_a, context * matrix_a, context * matrix_b), dim=-1)
        return self.ffn(fusion)


class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value, key_mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # [bs * nh, ql, dim] * [bs * nh, dim, kl] -> [bs * nh, ql, kl]
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1)  # [bs * nh, 1, kl]
            scores = scores.masked_fill(key_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical visual_layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CondAttention(nn.Module):
    def __init__(self, model_dim, head_num):
        super().__init__()
        self.injection_key = nn.Linear(model_dim, model_dim)
        self.injection_query = nn.Linear(model_dim, model_dim)
        self.attention = MultiheadAttention(model_dim, head_num)

    def forward(self, query, key, value, condition, key_padding_mask):
        """
        :param query: (L, B, D)
        :param key: (L, B, D)
        :param value: (L, B, D)
        :param key_padding_mask: (B, L)
        :param condition: (B, D) or None
        :return:
        """
        if condition is not None:
            query_weight = 1 + torch.sigmoid(self.injection_query(condition.unsqueeze(0)))
            key_weight = 1 + torch.sigmoid(self.injection_key(condition.unsqueeze(0)))
            query, key = query_weight * query, key_weight * key
        return self.attention(query, key, value, key_padding_mask=key_padding_mask)



