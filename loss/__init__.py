import math
import sys

import torch
from torch import nn

from loss.image_grounding import calc_reg_loss, calc_gIoU_loss, calc_cIoU_loss, calc_dIoU_loss, calc_iou_loss, \
    calc_heatmap_loss
from loss.embedding import calc_embedding_loss
import torch.nn.functional as F

from loss.info_nce import InfoNCE
from loss.vq_vae import calc_vq_loss, calc_straight_vq_loss, calc_align_loss
from utils.text_processor import bert_input2output


def calc_map_loss(ig_pred, ig_target, ac_audio, ac_text, config):
    """
    :param ac_text: [bs, dim]
    :param ac_audio: [bs, dim]
    :param ig_pred: [bs, 13, 13]
    :param ig_target: [bs, 416, 416]
    :param config: configuration about loss
    :return:
    """
    map_loss = calc_heatmap_loss(ig_pred, ig_target)
    emb_loss = calc_embedding_loss(ac_audio, ac_text)
    total_loss = config["map"] * map_loss + config["emb"] * emb_loss
    return total_loss, {
        "total": total_loss.item(),
        "emb": emb_loss.item(),
        "map": map_loss.item(),
    }


def calc_box_loss(ig_pred, ig_target, ac_audio, ac_text, config):
    """
    :param ac_text: [bs, dim]
    :param ac_audio: [bs, dim]
    :param ig_pred: [bs, 4] (x, y, w, h) (in [0, 1])
    :param ig_target: [bs, 4] （x, y, w, h) (in [0, 1])
    :param config: configuration about loss
    :return:
    """
    reg_loss = calc_reg_loss(ig_pred, ig_target)
    iou_loss = calc_iou_loss(ig_pred, ig_target, config["iou_type"])
    emb_loss = calc_embedding_loss(ac_audio, ac_text)
    total_loss = config["reg"] * reg_loss + config["iou"] * iou_loss + config["emb"] * emb_loss
    return total_loss, {
        "total": total_loss.item(),
        "emb": emb_loss.item(),
        "reg": reg_loss.item(),
        "iou": iou_loss.item()
    }


def calc_domain_loss(ac_domain, ig_domain):
    ac_gt = torch.ones_like(ac_domain)
    ig_gt = torch.zeros_like(ig_domain)
    ac_loss = F.binary_cross_entropy_with_logits(ac_domain, ac_gt)
    ig_loss = F.binary_cross_entropy_with_logits(ig_domain, ig_gt)
    return ac_loss + ig_loss


# def calc_overall_vq_loss(ig_pred, ig_target, ac_audio_core, ac_text_core, ac_audio, ac_text, ig_text,
#                          ac_audio_res, ac_text_res, ig_text_res, ac_domain, ig_domain, config):
#     # Image Grounding Loss
#     reg_loss = calc_reg_loss(ig_pred, ig_target)
#     iou_loss = calc_iou_loss(ig_pred, ig_target, config["iou_type"])
#     # Vector Quantized Loss
#     ig_text_vq_loss = calc_straight_vq_loss(ig_text_res["encoding"], ig_text_res["quantized"], config["lamb"])
#     ac_text_vq_loss = calc_straight_vq_loss(ac_text_res["encoding"], ac_text_res["quantized"], config["lamb"])
#     audio_vq_loss = calc_straight_vq_loss(ac_audio_res["encoding"], ac_audio_res["quantized"], config["lamb"])
#     # Reconstruction Loss
#     ac_gt_word, ig_gt_word = bert_input2output(ac_text), bert_input2output(ig_text)
#     ac_text_recon_loss = F.cross_entropy(ac_text_res["output"][:, :-1].transpose(-2, -1), ac_gt_word[:, 1:])
#     ig_text_recon_loss = F.cross_entropy(ig_text_res["output"][:, :-1].transpose(-2, -1), ig_gt_word[:, 1:])
#     audio_recon_loss = F.l1_loss(ac_audio_res["output"], ac_audio)
#     # Domain Loss
#     domain_loss = calc_domain_loss(ac_domain, ig_domain)
#     total_loss = (config["recon"]["text"] * (ac_text_recon_loss + ig_text_recon_loss) +
#                   config["recon"]["audio"] * audio_recon_loss +
#                   config["code"]["text"] * (ac_text_vq_loss + ig_text_vq_loss) +
#                   config["code"]["audio"] * audio_vq_loss +
#                   config["domain"] * domain_loss +
#                   config["reg"] * reg_loss + config["iou"] * iou_loss)
#     sim = F.cosine_similarity(ac_audio_core, ac_text_core, dim=-1).mean()
#     return total_loss, {
#         "total": total_loss.item(), "reg": reg_loss.item(), "iou": iou_loss.item(), "sim": sim.item(),
#         "ac_tr": ac_text_recon_loss.item(), "ig_tr": ig_text_recon_loss.item(), "ac_ar": audio_recon_loss.item(),
#         "ig_tq": ig_text_vq_loss.item(), "ac_tq": ac_text_vq_loss.item(), "ac_aq": audio_vq_loss.item(),
#         "adv": domain_loss.item()
#     }

def calc_contrastive_loss(audio, text, weight=None):
    if weight is None:
        return InfoNCE(temperature=0.1)(audio, text)
    else:
        return (InfoNCE(temperature=0.1, reduction='none')(audio, text) * weight.unsqueeze(-1)).mean()


def calc_audio_recon_loss(output, target, mask, weight=None):
    # (B, L, D), (B, L, D), (B, L)
    if weight is not None:
        return (F.l1_loss(output, target, reduction='none').mean(-1) * weight.unsqueeze(-1)).masked_select(mask).mean()
    else:
        return (F.l1_loss(output, target, reduction='none').mean(-1)).masked_select(mask).mean()


def calc_text_recon_loss(output, target, mask, weight=None):
    # (B, L, D), (B, L), (B, L, D)
    if weight is not None:
        return (F.cross_entropy(output.transpose(-2, -1), target, reduction='none') *
                weight.unsqueeze(-1)).masked_select(mask).mean()
    else:
        return (F.cross_entropy(output.transpose(-2, -1), target, reduction='none')).masked_select(mask).mean()


def calc_overall_vq_loss(ig_pred, ig_target, ac_audio, ac_text, ig_text, config, weight, epoch):
    ig_weight, ac_weight = weight["ig"], weight["ac"]
    # avg_ig, avg_ac = ig_domain.sigmoid().mean(), ac_domain.sigmoid().mean()
    # Image Grounding Loss
    reg_loss = calc_reg_loss(ig_pred, ig_target, ig_weight)
    iou_loss = calc_iou_loss(ig_pred, ig_target, config["iou_type"], ig_weight)
    # Alignment Loss
    # align_loss = calc_align_loss(ac_audio["probs"], ac_text["probs"])
    # Reconstruction Loss
    ig_text_recon = calc_text_recon_loss(ig_text["output"], ig_text["target"], ig_text["mask"])
    ac_text_recon = calc_text_recon_loss(ac_text["output"], ac_text["target"], ac_text["mask"], ac_weight)
    ac_audio_recon = calc_audio_recon_loss(ac_audio["output"], ac_audio["target"], ac_audio["mask"], ac_weight)
    # Domain Loss
    # domain_loss = calc_domain_loss(ac_domain, ig_domain)  # 越高越倾向于是audio-related
    # warmstart_adv = 2 / (1 + math.exp(-epoch / 5.)) - 1
    # domain_loss = calc_domain_loss(ac_domain_pro, ig_domain_pro)
    # Contrastive Loss
    contrastive_loss = calc_contrastive_loss(ac_audio["encoding"], ac_text["encoding"], ac_weight)
    # Extra Loss
    extra_loss = ((ac_audio["loss"]).mean() + (ac_text["loss"]).mean() + (ig_text["loss"]).mean())
    total_loss = (
        config["reg"] * reg_loss +
        config["iou"] * iou_loss +
        config["recon"] * (ig_text_recon + ac_text_recon + ac_audio_recon) / 3.0 +
        # config["domain"] * domain_loss +
        config["contrastive"] * contrastive_loss +
        config["extra"] * extra_loss
    )
    # assert torch.isnan(total_loss).sum() == 0, f"REG: {reg_loss}, IOU: {iou_loss}, IG TEXT: {ig_text_recon}," \
    #                                            f"AC TEXT: {ac_text_recon}, AC AUDIO: {ac_audio_recon}," \
    #                                            f"DOMAIN: {domain_loss}, CONTRAST: {contrastive_loss}," \
    #                                            f"EXTRA: {extra_loss}"
    sim = F.cosine_similarity(ac_audio["core"], ac_text["core"], dim=-1).mean()
    return total_loss, {
        "total": total_loss.item(), "reg": reg_loss.item(), "iou": iou_loss.item(), "sim": sim.item(),
        "ig_text": ig_text_recon.item(), "ac_text": ac_text_recon.item(), "ac_audio": ac_audio_recon.item(),
        "contrast": contrastive_loss.item(), "extra": extra_loss.item(), # "domain": domain_loss.item(),
        "ac_ap": ac_audio["perplexity"].mean().item(), "ac_tp": ac_text["perplexity"].mean().item(),
        "ig_tp": ig_text["perplexity"].mean().item()
    }


def calc_total_loss(*inputs, **kwargs):
    return calc_overall_vq_loss(*inputs, **kwargs)
    # return calc_box_loss(*inputs, **kwargs)
