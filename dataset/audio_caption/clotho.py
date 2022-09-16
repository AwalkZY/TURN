from dataset.audio_caption.base import AudioCaptionBase


class Clotho(AudioCaptionBase):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
