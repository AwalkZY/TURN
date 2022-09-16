import os
import sys

import torch
from torch.utils.data import Dataset
import torchaudio
from PIL import Image
import soundfile as sf
from scipy import signal
import numpy as np
from torchvision import transforms as T

from utils.accessor import load_json


class AudioLocalizationBase(Dataset):
    def __init__(self, data_root, anno_path, image_size, sample_rate, duration, n_fft, n_mels, **kwargs):
        super().__init__()
        self.annotation = load_json(anno_path)
        # self.multi_box = load_json(os.path.join(data_root, "multi_box.json"))
        # self.keys = [item for item in list(self.annotation.keys()) if item not in self.multi_box]
        self.keys = list(self.annotation.keys())
        self.audio_root = os.path.join(data_root, "audios")
        self.image_root = os.path.join(data_root, "images")
        self.sample_rate = sample_rate
        self.wave_length = duration * self.sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.image_size = image_size
        self.image_transform = T.Compose([
            T.Resize(image_size, Image.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_pool = {}
        self.audio_pool = {}
        self.dataset_name = None
        self.image2audio = None
        self.image2audio_path = os.path.join(data_root, "image2audio.json")

    def __len__(self):
        return len(self.keys)

    def _audio_transform(self, waveform, sample_rate):
        waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        if waveform.size(1) < self.wave_length:
            n = int(self.wave_length / waveform.size(1)) + 1
            waveform = torch.tile(waveform, (1, n))
        waveform = waveform[:, :self.wave_length].mean(0, keepdim=True)
        waveform[waveform > 1.] = 1.
        waveform[waveform < -1.] = -1.
        mel_spec = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                        n_mels=self.n_mels)(waveform)
        return mel_spec.transpose(-2, -1).squeeze(0)

    def _image_transform(self, image):
        return self.image_transform(image)

    def __getitem__(self, index):
        image_id = self.keys[index]
        audio_id = self.image2audio[image_id] if self.dataset_name == "MUSIC" else image_id
        boxes = self.annotation[image_id]
        gt_map = self._box2map(boxes)
        if image_id in self.image_pool and audio_id in self.audio_pool:
            image = self.image_pool[image_id]
            audio = self.audio_pool[audio_id]
        else:
            if self.dataset_name == "MUSIC":
                image = Image.open(os.path.join(self.image_root, image_id[:11], image_id + ".jpg")).convert("RGB")
                audio = torch.load(os.path.join(self.audio_root, audio_id[:11], audio_id + ".pt"))
            else:
                image = Image.open(os.path.join(self.image_root, image_id + ".jpg")).convert("RGB")
                waveform, sample_rate = torchaudio.load(os.path.join(self.audio_root, audio_id + ".wav"))
                audio = self._audio_transform(waveform, sample_rate)
            image = self._image_transform(image)
            self.image_pool[image_id] = image
            self.audio_pool[audio_id] = audio
        image_mask = torch.zeros(*image.size()[-2:])
        return image, image_mask, audio, gt_map, image_id

    def _box2map(self, boxes):
        raise NotImplementedError
