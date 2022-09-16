from dataset.audio_caption.base import AudioCaptionBase


class AudioCaps(AudioCaptionBase):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
