import sys
import numpy as np
import torch

from utils.container import metricsContainer
from utils.helper import move_to_cuda, move_to_cpu
from utils.metrics import GroundingMetrics
from utils.processor import tuple2dict
from utils.detection_transforms import box_transform


def image_grounding(model, data_loader, display_interval):
    """
        :param display_interval: the interval of displaying the prediction and ground-truth
        :param data_loader: the data loader providing data
        :param model: the model to be evaluated
        :return: dict[float]
    """
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            batch = tuple2dict(batch, ["image_data", "image_mask", "word_data", "word_mask", "bbox"])
            batch = move_to_cuda(batch)
            output = model(**batch)
            # pred: [bs, 4] (x, y, w, h) (in [0, 1])
            # target: [bs, 4] （x, y, w, h) (in [0, 1])
            # prediction, target = pred2bbox(output["prediction"]), batch["bbox"]
            prediction = box_transform(output["prediction"], src_format="xywh", tgt_format="xyxy").clamp(min=0, max=1)
            target = box_transform(batch["bbox"], src_format="xywh", tgt_format="xyxy").clamp(min=0, max=1)
            batch_size = prediction.size(0)
            if batch_idx % display_interval == 0:
                print("-" * 20)
                chosen_idx = np.random.randint(batch_size)
                print("Pred: {}, Target: {}".format(prediction[chosen_idx], target[chosen_idx]))
                print("-" * 20)
                sys.stdout.flush()

            metricsContainer.update("top_1", GroundingMetrics.top_1_metric(prediction.cpu(), target.cpu()))
        return {
            "top_1": metricsContainer.calculate_average("top_1")
        }


def image_grounding_quality(model, data_loader):
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            batch = tuple2dict(batch, ["image_data", "image_mask", "word_data", "word_mask", "bbox"])
            batch = move_to_cuda(batch)
            output = model(**batch)
            image_data = move_to_cpu(batch["image_data"])
            batch_size = len(image_data)
            visual_attn = move_to_cpu(output["base_map"])
            result_size = (batch_size, 4, 4, 20 * 20)
            visual_attn = visual_attn.view(*result_size)
            torch.save({
                "image": image_data,
                "visual_attn": visual_attn
            }, "info.pth")
            break
