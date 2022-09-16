from yacs.config import CfgNode as CN


_c = CN()
_c.model_dim = 256
_c.text_size = 30
_c.image_size = 384
_c.dropout = 0.3
_c.sample_rate = 22050
_c.duration = 10
_c.image_dim = 256
_c.word_dim = 768
_c.n_fft = 2048
_c.n_mels = 128
_c.reproductive = False
_c.path_root = ""
_c.use_weight = True

_c.model = CN()
_c.model.name = "VQModel"
_c.model.model_dim = _c.model_dim
_c.model.dropout = _c.dropout

_c.model.img_trm_cfg = CN()
_c.model.img_trm_cfg.model_dim = _c.model_dim
_c.model.img_trm_cfg.head_num = 4
_c.model.img_trm_cfg.num_layers = 3
_c.model.img_trm_cfg.dim_feedforward = 4 * _c.model_dim
_c.model.img_trm_cfg.operation = "prod"
_c.model.img_trm_cfg.dropout = _c.dropout
_c.model.img_trm_cfg.activation = "gelu"

_c.model.aud_trm_cfg = CN()
_c.model.aud_trm_cfg.model_dim = _c.model_dim
_c.model.aud_trm_cfg.head_num = 4
_c.model.aud_trm_cfg.num_layers = 3
_c.model.aud_trm_cfg.dim_feedforward = 4 * _c.model_dim
_c.model.aud_trm_cfg.dropout = _c.dropout
_c.model.aud_trm_cfg.activation = "gelu"

_c.model.vq_cfg = CN()
_c.model.vq_cfg.dim = _c.model_dim
_c.model.vq_cfg.audio_dim = _c.n_mels
_c.model.vq_cfg.head_num = 8
_c.model.vq_cfg.layer_num = 6
_c.model.vq_cfg.code_size = 256
_c.model.vq_cfg.groups = 2
_c.model.vq_cfg.gumbel_temperature = 0.5
_c.model.vq_cfg.is_single = True
_c.model.vq_cfg.decay = 0.99
_c.model.vq_cfg.epsilon = 1e-5


_c.model.image_cfg = CN()
_c.model.image_cfg.image_dim = _c.image_dim
# _c.model.image_cfg.backbone = "resnet101"
_c.model.image_cfg.detr_model_weight = "./misc/detr-r50-e632da11.pth"
_c.model.image_cfg.detr_model = CN()
_c.model.image_cfg.detr_model.backbone = "resnet50"
_c.model.image_cfg.detr_model.dropout = _c.dropout
_c.model.image_cfg.detr_model.position_embedding = "sine"
_c.model.image_cfg.detr_model.train_backbone = True
_c.model.image_cfg.detr_model.hidden_dim = 256
_c.model.image_cfg.detr_model.nheads = 8
_c.model.image_cfg.detr_model.dim_feedforward = 2048
_c.model.image_cfg.detr_model.enc_layers = 6

_c.model.text_cfg = CN()
_c.model.text_cfg.text_dim = _c.word_dim
_c.model.text_cfg.max_len = _c.text_size
_c.model.text_cfg.head_num = 4
_c.model.text_cfg.layer_num = 6
_c.model.text_cfg.bert_model = _c.path_root + "grounding_base/bert-base-uncased/"  # "bert-base-uncased"


_c.model.audio_cfg = CN()
_c.model.audio_cfg.audio_dim = _c.n_mels
_c.model.audio_cfg.audio_len = 216

_c.dataset = CN()
_c.dataset.vg = CN()
_c.dataset.vg.name = "VidSTG"
_c.dataset.vg.data_path = _c.path_root + "vidor"
_c.dataset.vg.imsize = _c.image_size
_c.dataset.vg.query_len = _c.text_size
_c.dataset.vg.vocab_path = _c.path_root + "grounding_base/bert-base-uncased/bert-base-uncased-vocab.txt"
_c.dataset.vg.augment = CN()
_c.dataset.vg.augment.aug_scale = True
_c.dataset.vg.augment.aug_crop = True
_c.dataset.vg.augment.aug_blur = False
_c.dataset.vg.augment.aug_translate = True


_c.dataset.ig = CN()
_c.dataset.ig.name = "ReferDataset"
_c.dataset.ig.dataset = "gref"
_c.dataset.ig.use_weight = _c.use_weight
_c.dataset.ig.data_root = _c.path_root + "grounding_base"
_c.dataset.ig.split_root = _c.path_root + "grounding_base/data"
_c.dataset.ig.vocab_path = _c.path_root + "grounding_base/bert-base-uncased/bert-base-uncased-vocab.txt"
_c.dataset.ig.imsize = _c.image_size
_c.dataset.ig.augment = CN()
_c.dataset.ig.augment.aug_scale = True
_c.dataset.ig.augment.aug_crop = True
_c.dataset.ig.augment.aug_blur = False
_c.dataset.ig.augment.aug_translate = True
_c.dataset.ig.max_query_len = _c.model.text_cfg.max_len


_c.dataset.al = CN()
_c.dataset.al.val = CN()
_c.dataset.al.val.name = "SoundNet"
_c.dataset.al.val.data_root = _c.path_root + "soundnet/"
_c.dataset.al.val.anno_path = _c.dataset.al.val.data_root + "annotation.json"
_c.dataset.al.val.image_size = _c.image_size
_c.dataset.al.val.sample_rate = _c.sample_rate
_c.dataset.al.val.duration = _c.duration
_c.dataset.al.val.n_fft = _c.n_fft
_c.dataset.al.val.n_mels = _c.n_mels

_c.dataset.al.test = CN()
_c.dataset.al.test.name = "VGGSS"
_c.dataset.al.test.data_root = _c.path_root + "vggss/"
_c.dataset.al.test.anno_path = _c.dataset.al.test.data_root + "annotation.json"
_c.dataset.al.test.image_size = _c.image_size
_c.dataset.al.test.sample_rate = _c.sample_rate
_c.dataset.al.test.duration = _c.duration
_c.dataset.al.test.n_fft = _c.n_fft
_c.dataset.al.test.n_mels = _c.n_mels


_c.dataset.ac = CN()
_c.dataset.ac.name = "AudioCaps"
_c.dataset.ac.data_root = _c.path_root + "audiocaps"
_c.dataset.ac.use_weight = _c.use_weight
_c.dataset.ac.sample_rate = _c.sample_rate
_c.dataset.ac.duration = _c.duration
_c.dataset.ac.max_query_len = _c.text_size
_c.dataset.ac.vocab_path = _c.dataset.ig.vocab_path
_c.dataset.ac.n_fft = _c.n_fft
_c.dataset.ac.n_mels = _c.n_mels
_c.dataset.ac.augment = CN()
_c.dataset.ac.augment.time_drop_width = 64
_c.dataset.ac.augment.time_stripes_num = 2
_c.dataset.ac.augment.freq_drop_width = 8
_c.dataset.ac.augment.freq_stripes_num = 2
_c.dataset.ac.augment.mask_type = "zero_value"

# Train Config
_c.train = CN()
_c.train.ig_split = "train"
_c.train.batch_size = 64
_c.train.shuffle = True
_c.train.max_epoch = 30
_c.train.max_mask_ratio = 1
_c.train.min_mask_ratio = 0.5
_c.train.display_interval = 50
_c.train.saved_path = ""

# _c.val = CN()
# _c.val.batch_size = 96

# Test Config
_c.test = CN()
_c.test.ig_split = "testA"
_c.test.batch_size = 72
_c.test.display_interval = 20

_c.optimizer = CN()
_c.optimizer.lr = 1e-4
_c.optimizer.max_lr = 3e-4
_c.optimizer.momentum = 0.9
_c.optimizer.weight_decay = 1e-5
_c.optimizer.T_0 = 4
_c.optimizer.T_max = _c.train.max_epoch
_c.optimizer.T_mult = 2
_c.optimizer.gamma = 0.93
_c.optimizer.warmup_epoch = 1
_c.optimizer.use_warmup = True
_c.optimizer.optim_type = "Cosine"
_c.optimizer.loss_config = CN()
_c.optimizer.loss_config.reg = 1
_c.optimizer.loss_config.map = 1
_c.optimizer.loss_config.iou = 1
_c.optimizer.loss_config.emb = 1
_c.optimizer.loss_config.domain = 1
_c.optimizer.loss_config.align = 0.5
_c.optimizer.loss_config.extra = 1
_c.optimizer.loss_config.contrastive = 0.1
_c.optimizer.loss_config.recon = 0.1
_c.optimizer.loss_config.code = CN()
_c.optimizer.loss_config.code.text = 1
_c.optimizer.loss_config.code.audio = 1
_c.optimizer.loss_config.lamb = 0.25
_c.optimizer.loss_config.iou_type = "gIoU"


# _c.optimizer.loss_config.size = _c.image_size


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
