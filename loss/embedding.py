import random

import torch


def calc_embedding_loss(audio_agg, text_agg):
    """
    :param audio_agg: FloatTensor, in (B, D)
    :param text_agg: FloatTensor, in (B, D)
    :return:
    """
    batch_size = audio_agg.size(0)
    rand_offset = random.randint(0, batch_size - 2)
    neg_idx = (torch.arange(batch_size) + rand_offset) % batch_size
    neg_text, neg_audio = text_agg[neg_idx], audio_agg[neg_idx]
    pos_label = torch.ones(batch_size).to(audio_agg.device)
    neg_label = torch.zeros(batch_size).to(audio_agg.device)
    pos_loss = torch.cosine_embedding_loss(audio_agg, text_agg, pos_label).mean()
    neg_loss_1 = torch.cosine_embedding_loss(audio_agg, neg_text, neg_label).mean()
    neg_loss_2 = torch.cosine_embedding_loss(neg_audio, text_agg, neg_label).mean()
    return pos_loss + (neg_loss_1 + neg_loss_2) / 2.0

