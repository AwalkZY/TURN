from dataset.audio_localization.base import AudioLocalizationBase
import numpy as np

from utils.calculator import box2map


class VGGSS(AudioLocalizationBase):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _box2map(self, boxes):
        # inputs: list of [float * 4], 0-1
        return box2map(boxes, self.image_size, "VGGSS")
