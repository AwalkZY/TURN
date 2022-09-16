import os
import random
import sys
import warnings

import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer

from utils.calculator import sigmoid_norm, softmax_norm
from utils.text_processor import read_examples, convert_examples_to_features
from dataset.spec_augmentation import SpecAugmentation
from utils.accessor import load_json


class AudioCaptionBase(Dataset):
    def __init__(self, data_root, vocab_path, augment, max_query_len, n_fft, n_mels, use_weight=True,
                 sample_rate=22050, duration=10, **kwargs):
        super().__init__()
        self.annotation = load_json(os.path.join(data_root, "annotation.json"))
        missing_path = os.path.join(data_root, "missing.json")
        if os.path.exists(missing_path):
            self.missing = load_json(missing_path)
        else:
            self.missing = []
        self.keys = [item for item in list(self.annotation.keys()) if item not in self.missing]
        self.use_weight = use_weight
        weight_path = os.path.join(data_root, "weight.pt")
        if self.use_weight and not os.path.exists(weight_path):
            raise FileNotFoundError
        if self.use_weight:
            self.weight = softmax_norm(torch.tensor(torch.load(weight_path)), reverse=True)
            # WARNING: score越低，越倾向于是image相关的item
        else:
            self.weight = torch.ones(len(self.keys))
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.audio_path = os.path.join(data_root, "audios")
        self.query_len = max_query_len
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.spec_augment = SpecAugmentation(**augment)
        self.sample_rate = sample_rate
        self.wave_length = sample_rate * duration
        self.audio_pool = {}

    def __len__(self):
        return len(self.keys)

    def _audio_transform(self, waveform, sample_rate):
        waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        if waveform.size(1) < self.wave_length:
            n = int(self.wave_length / waveform.size(1)) + 1
            waveform = torch.tile(waveform, (1, n))
        waveform = waveform[:, :self.wave_length]
        waveform[waveform > 1.] = 1.
        waveform[waveform < -1.] = -1.
        mel_spec = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                        n_mels=self.n_mels)(waveform)
        mel_spec = torch.log(mel_spec + 1e-7)
        mel_spec = mel_spec.transpose(-2, -1).unsqueeze(1)
        mel_spec = self.spec_augment(mel_spec).squeeze(1).squeeze(0)
        return mel_spec

    def __getitem__(self, index):
        file_id = self.keys[index]
        if file_id in self.audio_pool:
            waveform, sample_rate = self.audio_pool[file_id]
        else:
            audio_id = file_id[:-3] if file_id[-3:-1] == "##" else file_id
            waveform, sample_rate = torchaudio.load(os.path.join(self.audio_path, audio_id + ".wav"))
            self.audio_pool[file_id] = (waveform, sample_rate)
        mel_spec = self._audio_transform(waveform, sample_rate)
        caption = random.choice(self.annotation[file_id]).lower()
        examples = read_examples(caption, index)
        features = convert_examples_to_features(examples=examples, seq_length=self.query_len,
                                                tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        return mel_spec, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), index

