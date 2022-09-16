import sys

import torch
import torch.nn.functional as F

from utils.text_processor import bert_input2output


def calc_vq_loss(inputs, quantized, lamb):
    e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
    q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
    loss = q_latent_loss + lamb * e_latent_loss
    return loss

def calc_straight_vq_loss(inputs, quantized, lamb):
    return lamb * F.mse_loss(quantized.detach(), inputs)

def calc_perplexity(probs):
    avg_probs = torch.mean(probs, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
    return perplexity


def calc_align_loss(prob_1, prob_2, lamb=0.5):
    # Both in (B, G, S)
    prob_1, prob_2 = prob_1.flatten(0, 1)[:, None, :], prob_2.flatten(0, 1)[None, :, :]
    size = len(prob_1)
    prob_1 = prob_1.repeat_interleave(dim=1, repeats=size)
    prob_2 = prob_2.repeat_interleave(dim=0, repeats=size)
    # (B * G, 1, S) (1, B * G, S)
    log_prob_mean = ((prob_1 + prob_2) / 2.0).log()
    sim = 1 - (F.kl_div(log_prob_mean, prob_1, reduction='none') +
               F.kl_div(log_prob_mean, prob_2, reduction='none')).sum(-1) / 2.0 / size
    # (B * G, B * G)
    assert ((sim >= 0) * (sim <= 1)).all()
    target = torch.eye(len(sim)).to(sim.device) * lamb
    return F.mse_loss(sim, target)


# def calc_vqvae_loss(word_data, audio_data, text_output, audio_output, text_encoding,
#                     audio_encoding, text_quantized, audio_quantized, lamb=0.25, **kwargs):
#     text_vq_loss = calc_vq_loss(text_encoding, text_quantized, lamb)
#     audio_vq_loss = calc_vq_loss(audio_encoding, audio_quantized, lamb)
#     gt_word = bert_input2output(word_data)
#     text_recon_loss = F.cross_entropy(text_output[:, :-1].transpose(-2, -1), gt_word[:, 1:])
#     audio_recon_loss = F.l1_loss(audio_output, audio_data)
#     total_loss = text_vq_loss + audio_vq_loss + text_recon_loss + audio_recon_loss
#     return total_loss, {
#         "text_vq": text_vq_loss.item(),
#         "audio_vq": audio_vq_loss.item(),
#         "text_recon": text_recon_loss.item(),
#         "audio_recon": audio_recon_loss.item()
#     }

# def calc_vqvae_loss(word_data, audio_data, text_output, audio_output, text_encoding,
#                     audio_encoding, text_quantized, audio_quantized, lamb=0.25, **kwargs):
#     text_vq_loss = calc_vq_loss(text_encoding, text_quantized, lamb)
#     audio_vq_loss = calc_vq_loss(audio_encoding, audio_quantized, lamb)
#     gt_word = bert_input2output(word_data)
#     text_recon_loss = F.cross_entropy(text_output[:, :-1].transpose(-2, -1), gt_word[:, 1:])
#     audio_recon_loss = F.l1_loss(audio_output, audio_data)
#     total_loss = text_vq_loss + audio_vq_loss + text_recon_loss + audio_recon_loss
#     return total_loss, {
#         "text_vq": text_vq_loss.item(),
#         "audio_vq": audio_vq_loss.item(),
#         "text_recon": text_recon_loss.item(),
#         "audio_recon": audio_recon_loss.item()
#     }