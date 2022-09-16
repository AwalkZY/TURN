import os
import random
import sys
import time

from comet_ml import Experiment
import numpy as np
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

import dataset
import models
from criteria.audio_localization import audio_localization
from loss import calc_total_loss
from optimizers.lr_schedulers.warmup_scheduler import GradualWarmupScheduler
from utils.container import metricsContainer
from utils.helper import move_to_cuda
from utils.processor import tuple2dict
from utils.timer import Timer


class MainRunner:
    def __init__(self, config, use_comet):
        print("Initialization Start.")
        self.config = config
        self.use_comet = use_comet
        self._init_misc()
        self._init_dataset(config.dataset)
        self._init_model(config.model)
        self._init_optimizer(config.optimizer)
        print("Initialization End.")

    def _init_dataset(self, dataset_config):
        self.ig_dataset_cfg = dataset_config.ig
        self.ac_dataset_cfg = dataset_config.ac
        self.al_val_cfg = dataset_config.al.val
        self.al_test_cfg = dataset_config.al.test
        self.ig_train = getattr(dataset, self.ig_dataset_cfg.name, None)(**self.ig_dataset_cfg)
        self.ac_train = getattr(dataset, self.ac_dataset_cfg.name, None)(**self.ac_dataset_cfg)
        self.al_val = getattr(dataset, self.al_val_cfg.name, None)(**self.al_val_cfg)
        self.al_test = getattr(dataset, self.al_test_cfg.name, None)(**self.al_test_cfg)
        print("IG: {} samples, AC: {} samples, AL: {}/{} samples".format(len(self.ig_train), len(self.ac_train),
                                                                         len(self.al_val), len(self.al_test)))
        ac_size = int(self.train_config.batch_size * len(self.ac_train) / len(self.ig_train))
        self.ig_sampler = WeightedRandomSampler(weights=self.ig_train.weight,
                                                num_samples=len(self.ig_train),
                                                replacement=True)
        self.ac_sampler = WeightedRandomSampler(weights=self.ac_train.weight,
                                                num_samples=len(self.ac_train),
                                                replacement=True)
        self.ig_train_loader = DataLoader(self.ig_train, batch_size=self.train_config.batch_size,
                                          sampler=self.ig_sampler, num_workers=8, pin_memory=True)
        self.ac_train_loader = DataLoader(self.ac_train, batch_size=ac_size, sampler=self.ac_sampler,
                                          num_workers=4, pin_memory=True)
        self.al_val_loader = DataLoader(self.al_val, batch_size=self.test_config.batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True)
        self.al_test_loader = DataLoader(self.al_test, batch_size=self.test_config.batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)

    def _init_model(self, model_config):
        self.model = getattr(models, model_config.name, None)(**model_config)
        print("{:.2f}M".format(sum(p.numel() / 1e6 for p in self.model.parameters())))
        self.model = self.model.cuda(device=0)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

    def _export_log(self, epoch, total_step, batch_idx, lr, loss_meter, time_meter):
        msg = 'Epoch {}, Batch ({} / {}), lr = {:.5f}, '.format(epoch, batch_idx, len(self.ig_train_loader), lr)
        for k, v in loss_meter.items():
            msg += '{} = {:.4f}, '.format(k, v)
        remaining = len(self.ig_train_loader) - batch_idx
        msg += '{:.3f} s/batch ({}s left)'.format(time_meter, int(time_meter * remaining))
        print(msg)
        sys.stdout.flush()
        loss_meter.update({"epoch": epoch, "batch": total_step, "lr": lr})
        if self.use_comet:
            self.experiment.log_metrics(loss_meter)

    def _print_metrics(self, epoch, metrics, action, logging=True):
        msg = "{} Epoch {}".format(action, epoch)
        for k, v in metrics.items():
            msg += ', {} = {:.4f}'.format(k, v)
        print(msg)
        sys.stdout.flush()
        metrics.update({"epoch": epoch})
        if self.use_comet and logging:
            self.experiment.log_metrics(metrics)

    def _init_optimizer(self, optimizer_config):
        vis_param = self.model.module.image_extractor.parameters()
        # txt_param = self.model.module.text_extractor.parameters()
        other_param = [param for param in self.model.parameters()
                       if (param not in vis_param)]
        self.optimizer = torch.optim.AdamW([{'params': other_param},
                                            {'params': list(vis_param), 'lr': optimizer_config["lr"] / 10.0}],
                                           # {'params': list(txt_param), 'lr': optimizer_config["lr"] / 10.0}],
                                           lr=optimizer_config["lr"],
                                           weight_decay=optimizer_config["weight_decay"])
        if optimizer_config["optim_type"] == "OneCycle":
            self.sub_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=optimizer_config["max_lr"],
                                                                     steps_per_epoch=len(self.ig_train_loader),
                                                                     epochs=optimizer_config["T_max"])
        elif optimizer_config["optim_type"] == "Cosine":
            self.sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=optimizer_config["T_max"])
        elif optimizer_config["optim_type"] == "Expo":
            self.sub_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                        gamma=optimizer_config["gamma"])
        elif optimizer_config["optim_type"] == "CosineRestart":
            self.sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                      T_0=optimizer_config["T_0"],
                                                                                      T_mult=optimizer_config["T_mult"])
        else:
            raise NotImplementedError
        if optimizer_config["use_warmup"]:
            self.main_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                         total_epoch=optimizer_config["warmup_epoch"],
                                                         after_scheduler=self.sub_scheduler)
        else:
            self.main_scheduler = self.sub_scheduler

        self.loss_config = optimizer_config.loss_config

    def _init_misc(self):
        if self.config.reproductive:
            seed = 8
            random.seed(seed)
            np.random.seed(seed + 1)
            torch.manual_seed(seed + 2)
            torch.cuda.manual_seed(seed + 3)
            torch.cuda.manual_seed_all(seed + 4)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(self.config)
        if self.use_comet:
            self.experiment = Experiment()
            self.experiment.log_parameters(parameters=self.config)
        self.train_config = self.config.train
        self.test_config = self.config.test
        self.val_config = self.config.val if "val" in self.config else None
        self.model_saved_path = self.train_config["saved_path"]
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        self.initial_epoch = 0

    def _train_one_epoch(self, epoch, last_total_step):
        self.model.train()
        timer = Timer()
        for batch_idx, (ig_batch, ac_batch) in enumerate(zip(self.ig_train_loader,
                                                             self.ac_train_loader), 1):
            # timer.reset()
            self.optimizer.zero_grad()
            ig_batch = move_to_cuda(tuple2dict(ig_batch, ["image_data", "image_mask", "ig_text_data",
                                                          "ig_text_mask", "target", "index"]))
            ac_batch = move_to_cuda(tuple2dict(ac_batch, ["audio_data", "ac_text_data", "ac_text_mask", "index"]))
            max_mask_ratio, min_mask_ratio = self.train_config["max_mask_ratio"], self.train_config["min_mask_ratio"]
            mask_ratio = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * epoch / self.train_config["max_epoch"]
            # balance = 1 - min(2 * epoch / self.train_config["max_epoch"], 1)
            output = self.model(**{**ig_batch, **ac_batch}, mask_ratio=mask_ratio)
            loss, loss_items = calc_total_loss(
                ig_pred=output["prediction"], ig_target=ig_batch["target"],
                ac_audio=output["ac_audio"], ac_text=output["ac_text"], ig_text=output["ig_text"],
                # ac_domain=output["ac_domain"], ig_domain=output["ig_domain"],
                weight=output["weight"], config=self.loss_config, epoch=epoch
            )
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            self.main_scheduler.step(epoch + batch_idx / len(self.ig_train_loader))
            # self.main_scheduler.step()

            # update
            curr_lr = self.main_scheduler.get_last_lr()[0]
            time_interval = timer.elapsed_interval
            metricsContainer.update("loss", loss_items)
            metricsContainer.update("train_time", time_interval)

            if batch_idx % self.train_config.display_interval == 0:
                self._export_log(epoch, last_total_step + batch_idx, batch_idx, curr_lr,
                                 metricsContainer.calculate_average("loss"),
                                 metricsContainer.calculate_average("train_time"))
        if batch_idx % self.train_config.display_interval != 0:
            self._export_log(epoch, last_total_step + batch_idx, batch_idx, self.main_scheduler.get_last_lr()[0],
                             metricsContainer.calculate_average("loss"),
                             metricsContainer.calculate_average("train_time"))
        return batch_idx + last_total_step

    def eval(self, epoch, data, display_interval):
        """
        :param display_interval: decide the interval of log-printing
        :param data: decide which dataset is used to eval
        :param epoch: int
        :return:
        """
        self.model.eval()
        random_result = audio_localization(self.model, self.al_test_loader,
                                           display_interval=display_interval,
                                           use_random=True)
        self._print_metrics(epoch, random_result, data + "_Random", logging=False)
        val_result = audio_localization(self.model, self.al_val_loader,
                                        display_interval=display_interval,
                                        use_random=False)
        self._print_metrics(epoch, val_result, data + "_Eval", logging=False)
        test_result = audio_localization(self.model, self.al_test_loader,
                                         display_interval=display_interval,
                                         use_random=False)
        self._print_metrics(epoch, test_result, data + "_Test")
        return val_result, test_result

    # def quality_eval(self, data="Test"):
    #     self.model.eval()
    #     if data == "Test":
    #         data_loader = self.test_loader
    #     elif data == "Valid":
    #         data_loader = self.val_loader
    #     elif data == "Train":
    #         data_loader = self.train_loader
    #     else:
    #         raise NotImplementedError
    #     image_grounding_quality(self.model, data_loader)

    def train(self):
        total_step = 0
        best_result, best_criterion, best_epoch = (), float('-inf'), -1
        for epoch in range(self.initial_epoch, self.train_config["max_epoch"]):
            saved_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))
            if self.use_comet:
                with self.experiment.train():
                    total_step = self._train_one_epoch(epoch, total_step)
                self.save_model(saved_path, epoch)
                with self.experiment.test():
                    eval_result, test_result = self.eval(epoch, data="Test",
                                                         display_interval=self.test_config["display_interval"])
                if eval_result["AUC"] > best_criterion:
                    best_result = test_result
                    best_criterion = eval_result["AUC"]
                    best_epoch = epoch
            else:
                total_step = self._train_one_epoch(epoch, total_step)
                self.save_model(saved_path, epoch)
                eval_result, test_result = self.eval(epoch, data="Test",
                                                     display_interval=self.test_config["display_interval"])
                if eval_result["AUC"] > best_criterion:
                    best_result = test_result
                    best_criterion = eval_result["AUC"]
                    best_epoch = epoch
            print('=' * 60)
        print('-' * 120)
        print('Done.')
        print("Best Result:")
        self._print_metrics(best_epoch, best_result, "Result")

    def save_model(self, path, epoch):
        state_dict = {
            'epoch': epoch,
            'config': self.config,
            'model_parameters': self.model.state_dict()
        }
        torch.save(state_dict, path)
        print('save model to {}, epoch {}.'.format(path, epoch))

    def load_model(self, path):
        state_dict = torch.load(path)
        self.initial_epoch = state_dict['epoch']
        self.main_scheduler.step(self.initial_epoch)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        print('load model from {}, epoch {}.'.format(path, self.initial_epoch))

    # def qualitative_eval(decoder_layer, type_list, args_list):
    #     decoder_layer.model.eval()
    #     for eval_type, eval_args in zip(type_list, args_list):
    #         retrieval_result = qualitative_image_grounding(decoder_layer.model, decoder_layer.test_loader, **eval_args)
    #         save_json(retrieval_result, eval_args["filename"])
