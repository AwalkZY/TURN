# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.
Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""
import os.path as osp
import sys
from PIL import Image
import cv2
# import h5py
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.text_processor import read_examples, convert_examples_to_features

sys.path.append('.')
import utils
import re
import utils.detection_transforms as T
from transformers import BertTokenizer

# from transformers import BertTokenizer,BertModel

sys.modules['utils'] = utils

cv2.setNumThreads(0)


class DatasetNotFoundError(Exception):
    pass


input_transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

none_transform = ToTensor()


def make_transforms(image_size, aug_scale, aug_crop, aug_blur, aug_translate, split, one_stage=False):
    if one_stage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    if split == 'train':
        scales = []
        if aug_scale:
            for i in range(7):
                scales.append(image_size - 32 * i)
        else:
            scales = [image_size]

        if aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=aug_blur),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(size=image_size, aug_translate=aug_translate)
        ])

    if split in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([image_size]),
            T.ToTensor(),
            T.NormalizeAndPad(size=image_size),
        ])

    raise ValueError(f'unknown {split}')


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
    }

    def __init__(self, data_root, vocab_path, augment=None, split_root='data', dataset='referit', imsize=640,
                 return_idx=False, testmode=False, split='train', max_query_len=40, **kwargs):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root  # 'data'
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len  # 40
        self.transform = make_transforms(image_size=imsize, split=split, one_stage=False, **augment) \
            if augment is not None else none_transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.augment = augment
        self.return_idx = return_idx
        # self.image_pool = {}

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:  # refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)  # 存放annotation
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError('Dataset {0} does not have split {1}'.format(self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        # print(osp.join(self.split_root, self.dataset))
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        # box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        # if img_file in self.image_pool:
        #     img = self.image_pool[img_file]
        # else:
        #     img_path = osp.join(self.im_dir, img_file)
        #     img = Image.open(img_path).convert("RGB")
        #     self.image_pool[img_file] = img

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']

        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            return img, np.array(img_mask), np.array(word_id, dtype=int), \
                   np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
