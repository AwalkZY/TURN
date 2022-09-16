import torch
from torch import nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from utils.helper import masked_operation
from utils.text_processor import bert_input2output

"""
基于生成和重构的方案（似乎生成和重构并不重要）
SingleQuantized方法： 
    使用并列的Encoder-Decoder架构，将整个文本编码到一个单个的vector，然后以该vector为memory进行解码
MultiQuantized方法：
    使用级联的Encoder-Decoder架构，将整个文本编码到指定的隐空间，隐空间内进行VectorQuantize，然后继续向后解码预测结果
"""

"""
    Comments: Microsoft & Adobe 共同采用的方案都是：
        使用一个单独的token编码整个语句的语义 (sentence embedding)，使用该token完成*生成*
    Microsoft: 直接把z加到query、key和value上
    Adobe: 复制sentence embedding形成序列，作为attention的memory序列
"""


class QuantizedText(nn.Module):
    def __init__(self, dim, head_num, layer_num, codebook, is_single=True):
        super().__init__()
        self.dim = dim
        self.is_single = is_single
        self.encoder_cfg = BertConfig(hidden_size=dim, num_hidden_layers=layer_num, num_attention_heads=head_num)
        self.decoder_cfg = BertConfig(hidden_size=dim, num_hidden_layers=layer_num, num_attention_heads=head_num,
                                      add_cross_attention=True, is_decoder=True)
        # self.encoder = BertModel.from_pretrained(pretrained_path)
        self.encoder = BertModel(self.encoder_cfg)
        self.decoder = BertModel(self.decoder_cfg)
        self.codebook = codebook
        self.projector = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1), nn.BatchNorm1d(dim),
        )
        self.out_linear = nn.Linear(self.decoder_cfg.hidden_size, self.decoder_cfg.vocab_size)
        self.bind_weight()

    def bind_weight(self):
        self.out_linear.weight = self.encoder.get_input_embeddings().weight
        self.decoder.set_input_embeddings(self.encoder.get_input_embeddings())

    def random_mask(self, word_data, mask_ratio):
        real_word = (word_data != 101) * (word_data != 102) * (word_data != 0)  # [CLS], [SEP], [PAD]
        dice = torch.rand(word_data.size()).to(word_data.device)
        dice = dice.masked_fill(real_word.logical_not(), 2)
        mask = torch.logical_or(dice <= mask_ratio, dice == dice.min(-1, keepdim=True)[0]) * real_word
        assert (mask.sum(-1) != 0).all(), word_data
        masked_input = word_data.masked_fill(mask, 103)
        return masked_input, mask

    def single_quantize(self, word_data, word_mask):
        encodings = self.encoder(input_ids=word_data, token_type_ids=None, attention_mask=word_mask,
                                 output_hidden_states=True).last_hidden_state  # (B, L, D)
        encoding = masked_operation(encodings, word_mask, dim=1, operation="mean")  # (B, D)
        encoding = self.projector(encoding.unsqueeze(-1)).squeeze(-1)
        core, extra_loss, perplexity = self.codebook(encoding)
        return core, encoding, extra_loss, perplexity

    def multi_quantize(self, word_data, word_mask):
        encodings = self.encoder(input_ids=word_data, token_type_ids=None, attention_mask=word_mask,
                                 output_hidden_states=True).last_hidden_state  # (B, L, D)
        encodings = self.projector(encodings.transpose(-2, -1)).transpose(-2, -1)
        encoding = masked_operation(encodings, word_mask, dim=1, operation="mean")
        core, extra_loss, perplexity = self.codebook(encodings)
        core = masked_operation(core, word_mask, dim=1, operation="mean")
        return core, encoding, extra_loss, perplexity

    def quantize(self, word_data, word_mask):
        return self.single_quantize(word_data, word_mask) if self.is_single else self.multi_quantize(word_data,
                                                                                                     word_mask)

    def reconstruction(self, quantized, word_data, word_mask, mask_ratio):
        masked_word, mask = self.random_mask(word_data, mask_ratio)
        output = self.decoder(input_ids=masked_word, token_type_ids=None, attention_mask=word_mask,
                              output_hidden_states=True,
                              encoder_hidden_states=quantized.unsqueeze(1)).last_hidden_state
        output = self.out_linear(output)
        return output, mask

    def forward(self, word_data, word_mask, mask_ratio):
        """
        :param word_data: in (bs, max_len)
        :param word_mask: in (bs, max_len)
        :param mask_ratio: float (0, 1)
        :return:
        """
        """
            Comments: Microsoft & Adobe 共同采用的方案都是：
                使用一个单独的token编码整个语句的语义 (sentence embedding)，使用该token完成重构（自回归式重构）
            Microsoft: 直接把z加到query、key和value上
            Adobe: 复制sentence embedding形成序列，作为attention的memory序列
        """
        core, encoding, extra_loss, perplexity = self.quantize(word_data, word_mask)
        if mask_ratio != -1:
            output, mask = self.reconstruction(core, word_data, word_mask, mask_ratio)
        else:
            output, mask = None, None
        return core, encoding, extra_loss, output, mask, perplexity
