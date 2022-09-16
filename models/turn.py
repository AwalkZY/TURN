import torch
from torch import nn
from models.model_zoo.detr import build_detr, load_weights

from models.modules.condition_transformer import ConditionTransEncoder
from models.modules.vq_vae import SingleVQ
from utils.processor import tuple2dict


class VisualPredictor(nn.Module):
    def __init__(self, model_dim, target_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.ReLU(),
            nn.Linear(model_dim, target_dim)
        )

    def forward(self, image_enc):
        raw_pred = self.predictor(image_enc)
        return raw_pred.sigmoid()
        # (bs, dim) -> (bs, target_dim)


class MainModel(nn.Module):
    def __init__(self, image_cfg, audio_cfg, img_trm_cfg, vq_cfg, model_dim, dropout, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.image_cfg = image_cfg
        self.audio_len = audio_cfg["audio_len"]
        self.transformer_cfg = img_trm_cfg
        # 0.1 Extract image features
        self.image_extractor = build_detr(image_cfg["detr_model"])
        self.image_extractor = load_weights(self.image_extractor, image_cfg["detr_model_weight"])
        # 0.2 Extract text features & Extract audio features
        # 1. Initialize Projectors into the same latent space
        self.image_projector = nn.Sequential(
            nn.Conv2d(image_cfg["image_dim"], model_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(model_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(model_dim),
            nn.LeakyReLU()
        )
        self.quantized_model = SingleVQ(**vq_cfg)
        # self.init_classifier = SequenceDiscriminator(self.model_dim)
        # self.domain_classifier = SingleDiscriminator(self.model_dim, need_grl=False)
        self.grounding_transformer = ConditionTransEncoder(**img_trm_cfg)
        self.image_predictor = VisualPredictor(model_dim, 4)

    def project_image(self, image_data, image_mask):
        image_feat, image_mask = self.image_extractor(image_data, image_mask)
        image_feat = self.image_projector(image_feat)
        # image_feat = F.normalize(image_feat, p=2, dim=1)
        return image_feat, image_mask

    def grounding(self, image_data, image_mask, condition):
        batch_size = image_data.size(0)
        image_token, image_mask = self.project_image(image_data, image_mask)
        image_token = image_token.permute(0, 2, 3, 1).reshape(batch_size, -1, self.model_dim)
        # (bs, dim, mw, mh) -> (bs, mw * mh, dim)
        result_feat = self.grounding_transformer(image_token, condition, image_mask)
        prediction = self.image_predictor(result_feat.mean(1))
        return prediction

    def forward(self, image_data, image_mask, ig_text_data=None, ig_text_mask=None,
                ac_text_data=None, ac_text_mask=None, audio_data=None, mask_ratio=0.0, use_random=False, **kwargs):
        """
        :param image_data: in (bs, 3, max_width, max_height)
        :param image_mask: in (bs, max_width, max_height)
        :param ig_text_data: in (bs, max_len)
        :param ig_text_mask: in (bs, max_len)
        :param ac_text_data: in (bs, max_len)
        :param ac_text_mask: in (bs, max_len)
        :param audio_data: in (bs, max_time, freq_bins)
        :param mask_ratio: float
        :return:
        """
        if self.training:
            ig_text = self.quantized_model.project_text(ig_text_data, ig_text_mask)
            audio, ac_text = self.quantized_model.cross_text_audio(audio_data, ac_text_data, ac_text_mask, mask_ratio)
            audio_dict = tuple2dict(audio, ["core", "encoding", "loss", "output", "mask", "perplexity"])
            ac_text_dict = tuple2dict(ac_text, ["core", "encoding", "loss", "output", "mask", "perplexity"])
            ig_text_dict = tuple2dict(ig_text, ["core", "encoding", "loss", "output", "mask", "perplexity"])
            audio_dict["target"], ac_text_dict["target"], ig_text_dict[
                "target"] = audio_data, ac_text_data, ig_text_data
            prediction = self.grounding(image_data, image_mask, ig_text_dict["core"])
            return {
                "prediction": prediction,  # (x, y, w, h)
                "weight": {"ig": None, "ac": None},
                "ac_text": ac_text_dict, "ig_text": ig_text_dict, "ac_audio": audio_dict,
            }
        else:
            core = self.quantized_model.project_audio(audio_data)[0]
            if use_random:
                core = torch.randn_like(core)
            prediction = self.grounding(image_data, image_mask, core)
            return {
                "prediction": prediction  # (x, y, w, h)
            }
