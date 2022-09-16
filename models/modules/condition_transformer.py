from torch import nn
from torch.nn.modules.transformer import _get_clones
import torch.nn.functional as F
from models.modules.attention import CondAttention
from models.modules.transformer import LayerScaleBlock, FeedForwardBlock, PositionEncoder


class ConditionEncoderLayer(nn.Module):
    def __init__(self, model_dim, head_num, dim_feedforward=2048,
                 dropout=0.1, activation="relu", need_pos=True, **kwargs):
        super().__init__()
        self.need_pos = need_pos
        self.self_attn = CondAttention(model_dim, head_num)
        self.pre_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 2)
        self.post_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 2)
        self.layer_scale_block = nn.ModuleList([LayerScaleBlock(model_dim, dropout)] * 2)
        self.feed_forward = FeedForwardBlock(model_dim, dim_feedforward, dropout, activation)
        self.pos_encoder = PositionEncoder(model_dim)

    def forward(self, inputs, condition, padding_mask=None, pos=None):
        """
        :param inputs: Tensor, (B, L, D), Default: Batch First
        :param condition: Tensor, (B, D) or None
        :param padding_mask: ByteTensor, (B, L), NOTICE: Invalid bit is True
        :param pos: Tensor, (B, L, D)
        :return:
        """
        if self.need_pos:
            value = self.pre_norm[0](inputs)
            query = key = value + (pos if pos is not None else self.pos_encoder(value.size(1)))
        else:
            query = key = value = self.pre_norm[0](inputs)
        new_inputs, self_attn_weight = self.self_attn(query=query.transpose(0, 1),
                                                      key=key.transpose(0, 1),
                                                      value=value.transpose(0, 1),
                                                      condition=condition,
                                                      key_padding_mask=padding_mask)
        inputs = self.layer_scale_block[0](inputs, self.post_norm[0](new_inputs.transpose(0, 1)))
        inputs = self.layer_scale_block[1](inputs, self.post_norm[1](self.feed_forward(self.pre_norm[1](inputs))))
        return inputs, self_attn_weight


class ConditionTransEncoder(nn.Module):
    def __init__(self, model_dim, head_num, num_layers, dim_feedforward, operation, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = ConditionEncoderLayer(model_dim, head_num, dim_feedforward,
                                              dropout, activation, need_pos=True)
        self.norm = nn.LayerNorm(model_dim)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.operation = operation

    def forward(self, src, cond, src_key_padding_mask=None, src_pos=None):
        if self.operation == "prod":
            output = F.normalize(src * cond.unsqueeze(1), dim=-1)
            for layer in self.layers:
                output, _ = layer(output, None, src_key_padding_mask, src_pos)
        elif self.operation == "gate":
            output = src
            for layer in self.layers:
                output, _ = layer(output, cond, src_key_padding_mask, src_pos)
        else:
            raise NotImplementedError
        return output if self.norm is None else self.norm(output)
