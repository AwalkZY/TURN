from dataset.audio_localization.base import AudioLocalizationBase
import numpy as np
import os
from utils.accessor import load_json
from utils.calculator import box2map


class MUSIC(AudioLocalizationBase):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.dataset_name = "MUSIC"
        self.image2audio = load_json(self.image2audio_path)

    def _box2map(self, boxes):
        # inputs: list of [float * 4], 0-1
        return box2map(boxes, self.image_size, "MUSIC")
