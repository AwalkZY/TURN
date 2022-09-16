import torch
from torch import nn

from models.model_zoo.detr.backbone import build_backbone
from models.model_zoo.detr.misc import NestedTensor
from models.model_zoo.detr.transformer_backbone import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer_backbone.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, img, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        # pos: position encoding
        features, pos = self.backbone(samples)  # pos:list, pos[-1]: [64, 256, 20, 20]

        src, mask = features[-1].decompose()  # src:[64, 256, 20, 20]  mask:[64, 20, 20]
        assert mask is not None
        # FIXME: Modified here.
        bs, _, h, w = src.size()
        src_input = self.input_proj(src).flatten(2).permute(2, 0, 1)  # [bs, c, h, w] -> [hw, bs, c]
        pos_input = pos[-1].flatten(2).permute(2, 0, 1)
        mask_input = mask.flatten(1)  # [bs, h * w]
        feat_out = self.transformer(src_input, mask_input, pos_input)
        out = feat_out.permute(1, 2, 0).view(bs, -1, h, w)
        return out, mask_input


def load_weights(model, load_path):
    # 加载DETR模型的Transformer Encoder层与backbone层ResNet50的预训练参数
    dict_trained = torch.load(load_path)['model']
    # new_list = list(model.state_dict().keys())
    # trained_list = list(dict_trained.keys())
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model


def build_detr(args):
    backbone = build_backbone(args)  # ResNet 50
    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
    )
    return model
