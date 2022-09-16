import math

import torch

from utils.detection_transforms import box_transform
import torch.nn.functional as F


def calc_reg_loss(output, target, weight=None):
    if weight is None:
        weight = torch.ones_like(target[:, 0])
    l1_loss = torch.nn.L1Loss(reduction='none')
    loss_x = l1_loss(output[:, 0], target[:, 0])
    loss_y = l1_loss(output[:, 1], target[:, 1])
    loss_w = l1_loss(output[:, 2], target[:, 2])
    loss_h = l1_loss(output[:, 3], target[:, 3])
    return ((loss_x + loss_y + loss_w + loss_h) * weight).mean()


def calc_cIoU_loss(boxes1, boxes2):
    bs = boxes1.size(0)
    w1, h1 = boxes1[:, 2], boxes1[:, 3]
    w2, h2 = boxes2[:, 2], boxes2[:, 3]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1, center_y1 = boxes1[:, 0], boxes1[:, 1]
    center_x2, center_y2 = boxes2[:, 0], boxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    return torch.sum(1 - cious) / bs


def calc_dIoU_loss(boxes1, boxes2):
    bs = boxes1.size(0)
    w1, h1 = boxes1[:, 2], boxes1[:, 3]
    w2, h2 = boxes2[:, 2], boxes2[:, 3]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1, center_y1 = boxes1[:, 0], boxes1[:, 1]
    center_x2, center_y2 = boxes2[:, 0], boxes2[:, 1]
    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    return torch.sum(1 - dious) / bs


def calc_gIoU_loss(boxes1, boxes2, weight=None):
    '''
    cal GIOU of two boxes or batch boxes
    '''
    if weight is None:
        weight = torch.ones_like(boxes1[:, 0])
    bs = boxes1.size(0)
    max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)  # 确保tensor的下限是0
    inter = inter[:, 0] * inter[:, 1]
    # 分别计算boxes1和boxes2的像素面积
    boxes1Area = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))
    boxes2Area = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))

    union_area = boxes1Area + boxes2Area - inter + 1e-5
    ious = inter / union_area

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_right_down = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose = torch.clamp((enclose_right_down - enclose_left_up), min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1] + 1e-5

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area  # GIOU的值在[-1,1]，越接近1越好

    # GIOU Loss
    assert gious.size() == weight.size()
    giou_loss = (((1 - gious) * weight).sum()) / bs

    return giou_loss


def calc_iou_loss(pred, target, iou_type, weight=None):
    if iou_type == "gIoU":
        pred_xyxy = box_transform(pred, src_format="xywh", tgt_format="xyxy").clamp(min=0, max=1)
        target_xyxy = box_transform(target, src_format="xywh", tgt_format="xyxy").clamp(min=0, max=1)
        iou_loss = calc_gIoU_loss(pred_xyxy, target_xyxy, weight)
    elif iou_type in ["cIoU", "dIoU"]:
        pred_cxcywh = box_transform(pred, src_format="xywh", tgt_format="cxcywh")
        target_cxcywh = box_transform(target, src_format="xywh", tgt_format="cxcywh")
        if iou_type is "cIoU":
            iou_loss = calc_cIoU_loss(pred_cxcywh, target_cxcywh)
        else:
            iou_loss = calc_dIoU_loss(pred_cxcywh, target_cxcywh)
    else:
        raise NotImplementedError
    return iou_loss

def calc_heatmap_loss(heatmap, target):
    """
    :param heatmap: in (bs, 13, 13)
    :param target: in (bs, 416, 416)
    :return:
    """
    map_size = heatmap.size()[1:]
    gt_map = F.adaptive_avg_pool2d(target.unsqueeze(1), output_size=map_size).squeeze(1)
    return F.binary_cross_entropy_with_logits(heatmap, gt_map)


def calc_grounding_loss(pred, target, config):
    """
    # :param base_map: in (nl * bs, bn, mv)
    :param pred: [bs, 4] (x, y, w, h) (in [0, 1])
    :param target: [bs, 4] （x, y, w, h) (in [0, 1])
    :param config: configuration about loss
    :return:
    """
    reg_loss = calc_reg_loss(pred, target)
    iou_loss = calc_iou_loss(pred, target, config["iou_type"])
    total_loss = config["reg"] * reg_loss + config["iou"] * iou_loss  # + config["attn"] * attn_loss
    return total_loss, {
        "total": total_loss.item(),
        "sum": (reg_loss + iou_loss).item(),
        "reg": reg_loss.item(),
        "iou": iou_loss.item()
    }
