import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel


class QuantizedAudio(nn.Module):
    def __init__(self, dim, input_dim, head_num, layer_num, codebook, is_single=True):
        super().__init__()
        self.dim = dim
        self.is_single = is_single
        self.encoder_cfg = BertConfig(hidden_size=dim, num_hidden_layers=layer_num, num_attention_heads=head_num)
        self.decoder_cfg = BertConfig(hidden_size=dim, num_hidden_layers=layer_num, num_attention_heads=head_num,
                                      add_cross_attention=True, is_decoder=True)
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
        )
        self.output_deconv = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(dim, input_dim, kernel_size=4, stride=2, padding=1)
        )
        self.codebook = codebook
        self.projector = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1), nn.BatchNorm1d(dim),
        )
        self.mask_embedding = nn.Parameter(torch.randn(input_dim))
        self.encoder = BertModel(self.encoder_cfg)
        self.decoder = BertModel(self.decoder_cfg)

    def random_mask(self, audio, mask_ratio):
        # audio: in (B, L, D_in)
        masked_audio = audio.clone()
        batch_size, max_len = audio.size(0), audio.size(1)
        mask = (torch.rand(batch_size, max_len) < mask_ratio).to(audio.device)
        masked_audio[mask] = self.mask_embedding.to(masked_audio.dtype)
        return masked_audio, mask
        # (B, L, D_in), (B, L)

    def multi_quantize(self, audio):
        # Deprecated!
        audio = self.input_conv(audio.transpose(-2, -1)).transpose(-2, -1)  # (B, L / 4, D)
        encodings = self.encoder(inputs_embeds=audio, output_hidden_states=True).last_hidden_state
        encodings = self.projector(encodings.transpose(-2, -1)).transpose(-2, -1)
        encoding = encodings.mean(1)  # [B, D]
        core, extra_loss, perplexity = self.codebook(encodings)
        core = core.mean(1)
        return core, encoding, extra_loss, perplexity

    def single_quantize(self, audio):
        audio = self.input_conv(audio.transpose(-2, -1)).transpose(-2, -1)  # (B, L / 4, D)
        encodings = self.encoder(inputs_embeds=audio, output_hidden_states=True).last_hidden_state
        encoding = encodings.mean(1)  # (B, D)
        encoding = self.projector(encoding.unsqueeze(-1)).squeeze(-1)
        core, extra_loss, perplexity = self.codebook(encoding)  # (B, D), (B, S)
        return core, encoding, extra_loss, perplexity

    def quantize(self, audio):
        return self.single_quantize(audio) if self.is_single else self.multi_quantize(audio)

    def reconstruction(self, quantized, audio, mask_ratio):
        masked_audio, mask = self.random_mask(audio, mask_ratio)
        masked_audio = self.input_conv(masked_audio.transpose(-2, -1)).transpose(-2, -1)  # (B, L / 4, D)
        output = self.decoder(inputs_embeds=masked_audio, output_hidden_states=True,
                              encoder_hidden_states=quantized.unsqueeze(1)).last_hidden_state
        output = self.output_deconv(output.transpose(-2, -1)).transpose(-2, -1)
        return output, mask

    def forward(self, audio, mask_ratio):
        """
        :param audio: in (B, L, D_in)
        :return:
        """
        core, encoding, extra_loss, perplexity = self.quantize(audio)
        output, mask = self.reconstruction(core, audio, mask_ratio)
        return core, encoding, extra_loss, output, mask
