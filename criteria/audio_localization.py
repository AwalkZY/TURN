import sys

import numpy as np
import sklearn

import torch

import torch.nn.functional as F

from utils.accessor import save_pickle, save_json
from utils.calculator import box2map
from utils.container import metricsContainer
from utils.helper import move_to_cuda
from utils.processor import tuple2dict
from utils.detection_transforms import box_transform


class Evaluator:
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer_map, gt_map):
        ciou = np.sum(infer_map * gt_map) / (np.sum(gt_map) + np.sum(infer_map * (gt_map == 0)))
        self.ciou.append(ciou)
        return ciou, np.sum(infer_map * gt_map), (np.sum(gt_map) + np.sum(infer_map * (gt_map == 0)))

    # def cal_CIOU(self, infer, gt_map, thres):
    #     infer_map = np.zeros_like(gt_map)
    #     infer_map[infer >= thres] = 1
    #     ciou = np.sum(infer_map * gt_map) / (np.sum(gt_map) + np.sum(infer_map * (gt_map == 0)))
    #     self.ciou.append(ciou)
    #     return ciou, np.sum(infer_map * gt_map), (np.sum(gt_map) + np.sum(infer_map * (gt_map == 0)))

    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou) >= 0.05 * i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        # print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou) >= 0.5)
        return ciou

    def clear(self):
        self.ciou = []


def audio_localization(model, data_loader, display_interval, use_random):
    """
        :param use_random: whether the query takes random vectors
        :param display_interval: the interval of displaying the prediction and ground-truth
        :param data_loader: the data loader providing data
        :param model: the model to be evaluated
        :return: dict[float]
    """
    evaluator = Evaluator()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            file_ids = batch[-1]
            batch = tuple2dict(batch[:-1], ["image_data", "image_mask", "audio_data", "gt_map"])
            gt_maps = batch["gt_map"]
            batch = move_to_cuda(batch)
            output = model(**batch, use_random=use_random)
            # pred: [bs, 4] (x, y, w, h) (in [0, 1])
            prediction = box_transform(output["prediction"], src_format="xywh", tgt_format="xyxy")
            batch_size = len(prediction)
            prediction = prediction.clamp(min=0, max=1).cpu().numpy()
            if batch_idx % display_interval == 0:
                chosen_idx = np.random.randint(batch_size)
            else:
                chosen_idx = -1
            for i, (pred_box, gt_map, file_id) in enumerate(zip(prediction, gt_maps, file_ids)):
                pred_map = box2map(pred_box[None], max_size=gt_map.size(-1))
                ciou, inter, union = evaluator.cal_CIOU(pred_map, gt_map.numpy())
                results.append((file_id, ciou, pred_box.tolist()))
                if i == chosen_idx:
                    print("-" * 20)
                    print("cIoU: {}".format(ciou))
                    sys.stdout.flush()
        auc = evaluator.cal_AUC()
        ciou = evaluator.final()
        save_json(sorted(results, key=lambda item: -item[1]), "eval_result.json")
        return {
            "cIoU": ciou, "AUC": auc
        }

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


# def audio_localization(model, data_loader, display_interval, dataset):
#     """
#         :param display_interval: the interval of displaying the prediction and ground-truth
#         :param data_loader: the data loader providing data
#         :param model: the model to be evaluated
#         :param dataset: the name of dataset
#         :return: dict[float]
#     """
#     evaluator = Evaluator()
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(data_loader, 1):
#             batch = tuple2dict(batch, ["image_data", "image_mask", "audio_data", "gt_map"])
#             gt_maps = batch["gt_map"]
#             batch = move_to_cuda(batch)
#             output = model(**batch)["prediction"].sigmoid() # pred: [bs, 13, 13]
#             batch_size = gt_maps.size(0)
#             if batch_idx % display_interval == 0:
#                 chosen_idx = np.random.randint(batch_size)
#             else:
#                 chosen_idx = -1
#             for batch_i in range(batch_size):
#                 pred_map = F.interpolate(output[batch_i].reshape(1, 1, *(output.size()[-2:])),
#                                          gt_maps.size()[-2:], mode='nearest').squeeze()
#                 # pred_map = pred_map.cpu().numpy()
#                 pred_map = (1 - normalize_img(-pred_map)).cpu().numpy()
#                 threshold = np.sort(pred_map.flatten())[int(pred_map.shape[0] * pred_map.shape[1] / 2)]
#                 pred_map[pred_map > threshold] = 1
#                 pred_map[pred_map < 1] = 0
#                 ciou, inter, union = evaluator.cal_CIOU(pred_map, gt_maps[batch_i].numpy(), 0.5)
#                 if batch_i == chosen_idx:
#                     print("-" * 20)
#                     print("cIoU: {}".format(ciou))
#                     sys.stdout.flush()
#
#         auc = evaluator.cal_AUC()
#         ciou = evaluator.final()
#         return {
#             "cIoU": ciou, "AUC": auc
#         }
