import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.losses import build_loss
from src.trainer.base_trainer import BaseTrainer
from src.utils.logger import IndentedLog, log_dict_items
from src.utils.path import PathHandler
from src.utils.registry import TRAINER_REGISTRY

from .optimizer import build_optimizer, build_scheduler


@dataclass
class ModelOutput:
    real_images: Tensor
    fake_images: Tensor
    bpp: Tensor
    qbpp: Tensor
    out_vq_latent: Tensor
    gt_vq_latent: Tensor
    out_vq_logits: Tensor
    gt_vq_indices: Tensor
    y_hat: Tensor
    vq_accuracy: Tensor
    other_outputs: Dict

    def __init__(self, output_dict: Dict) -> None:
        pop_keys = [
            "real_images",
            "fake_images",
            "bpp",
            "qbpp",
            "out_vq_latent",
            "gt_vq_latent",
            "out_vq_logits",
            "gt_vq_indices",
            "y_hat",
            "vq_accuracy",
        ]
        for k in pop_keys:
            v = output_dict.pop(k)
            setattr(self, k, v)

        self.other_outputs = output_dict


@TRAINER_REGISTRY.register()
class RateDistortionVqCodeTrainer(BaseTrainer):
    def __init__(
        self,
        opt,
    ) -> None:
        super().__init__(opt)

    def _set_optimizer_scheduler(self) -> None:
        self.comp_model.vq_model.requires_grad_(False)
        parameters_dict, aux_parameters_dict = self.comp_model.separete_aux_parameters()

        # set g_optimizer
        optim_opt = deepcopy(self.opt.optim)
        with IndentedLog(level="INFO", msg="building g_optimizer"):
            self.g_optimizer = build_optimizer(
                parameters_dict, optim_opt.g_optimizer
            )
            self.g_scheduler = build_scheduler(
                self.g_optimizer, optim_opt.g_scheduler
            )

        # set aux_optimizer
        if len(aux_parameters_dict) > 0:
            with IndentedLog(level="INFO", msg="building aux_optimizer"):
                self.aux_optimizer = build_optimizer(
                    aux_parameters_dict, optim_opt.aux_optimizer
                )
        else:
            self.logger.warn("aux_optimizer is NOT build.")
            self.aux_optimizer = None

        _clip_max_norm = self.opt.optim.get("clip_max_norm", None)
        log_dict_items({"clip_max_norm": _clip_max_norm}, level="INFO", indent=False)

    def _set_losses(self) -> None:
        loss_opt = deepcopy(self.opt.loss)

        self.distortion_loss = build_loss(
            loss_opt.distortion_loss, loss_name="distortion_loss"
        )
        self.rate_loss = build_loss(loss_opt.rate_loss, loss_name="rate_loss")
        self.perceptual_loss = build_loss(
            loss_opt.perceptual_loss, loss_name="perceptual_loss"
        ).to(self.device)

        ### VQ Code Loss
        self.code_distortion_loss = build_loss(
            loss_opt.code_distortion_loss, loss_name="code_distortion_loss"
        ).to(self.device)
        self.code_ce_loss = build_loss(
            loss_opt.code_ce_loss, loss_name="code_ce_loss"
        ).to(self.device)

    def run_comp_model(self, data_dict: Dict) -> ModelOutput:
        out_dict = self.comp_model.run_model(
            **data_dict,
            is_train=True,
            fix_entropy_models=False,
            run_vq_decoder=True,
        )
        return ModelOutput(out_dict)

    def optimize_parameters(
        self, current_iter: int, data_dict: dict
    ) -> Union[dict, None]:
        log_dict = {}

        ###################################################################
        #                             Train G
        ###################################################################
        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        # run model
        outputs = self.run_comp_model(data_dict)
        other_outputs = outputs.other_outputs

        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        # calculate losses
        g_loss_dict = {}

        g_loss_dict["rate"] = self.rate_loss(outputs.bpp)
        ## Image Loss
        g_loss_dict["distortion"] = self.distortion_loss(
            outputs.real_images, outputs.fake_images
        )
        g_loss_dict["perceptual"] = self.perceptual_loss(
            outputs.real_images, outputs.fake_images
        )

        ## Code Loss
        g_loss_dict["code_distortion"] = self.code_distortion_loss(
            outputs.gt_vq_latent, outputs.out_vq_latent
        )
        g_loss_dict["code_ce"] = self.code_ce_loss(
            outputs.out_vq_logits, outputs.gt_vq_indices
        )

        l_total: Tensor = sum(_v for _v in g_loss_dict.values())  # type: ignore

        # For stability
        if loss_anomaly := self.check_loss_nan_inf(l_total):
            self.logger.warning(
                f"iter{current_iter}: skipped because loss is {loss_anomaly}"
            )
            return None  # skip back-propagation part

        # back prop & update parameters
        l_total.backward()
        if self.opt.optim.get("clip_max_norm", None):
            nn.utils.clip_grad_norm_(
                self.comp_model.parameters(), self.opt.optim.clip_max_norm
            )

        self.g_optimizer.step()

        log_dict.update(g_loss_dict)
        log_dict["lr"] = self.g_optimizer.param_groups[0]["lr"]

        self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict["aux"] = self.optimize_aux_parameters()

        return log_dict
    
    def _validation(self) -> Dict:
        eval_df = self.comp_model.validation(self.eval_loader, max_sample_size=100)
        eval_dict = eval_df.drop('idx', axis=1).mean().to_dict()
        return eval_dict

    def optimize_aux_parameters(self):
        if self.aux_optimizer is None:
            return
        aux_loss = self.comp_model.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()
        return aux_loss

    def save(self, current_iter: int):
        # save model
        self.model_saver.save(
            {"comp_model": self.comp_model},
            "comp_model",
            current_iter,
            keep=True,
        )
        # save training_state
        optimizer_scheduler_dict = {"g_optimizer": self.g_optimizer}
        optimizer_scheduler_dict["g_scheduler"] = self.g_scheduler
        if self.aux_optimizer:
            optimizer_scheduler_dict["aux_optimizer"] = self.aux_optimizer
        self.model_saver.save(
            optimizer_scheduler_dict,
            "training_state",
            current_iter,
            keep=self.opt.get("keep_training_state", False),
        )

    def _load_checkpoint(
        self,
        exp_name: str,
        itr: int,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        new_g_lr: Optional[float] = None,
        strict: bool = True,
        **kwargs,
    ) -> None:
        ## get checkpoint path
        path_handler = PathHandler(self.opt.path.ckpt_root, exp_name)
        exp_path_dict = path_handler.get_exp_path_dict()
        model_ckpt_path = path_handler.get_ckpt_path("comp_model", itr)
        assert os.path.exists(model_ckpt_path)

        if load_optimizer:
            optim_ckpt_path = path_handler.get_ckpt_path("training_state", itr)
            assert os.path.exists(optim_ckpt_path)
        else:
            optim_ckpt_path = None
            self.logger.warn("optimizer is not loaded")

        log_dict_items(
            {
                "model_ckpt_path": model_ckpt_path,
                "optim_ckpt_path": optim_ckpt_path,
            },
            level="INFO",
            indent=False,
        )

        model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        out = self.comp_model.load_state_dict(
            model_ckpt["comp_model"], strict=strict
        )
        self.logger.debug(f'comp_model.load_state_dict: "{out}"')

        if not load_optimizer:
            return

        training_state_ckpt = torch.load(
            optim_ckpt_path, map_location=self.device
        )
        self.g_optimizer.load_state_dict(training_state_ckpt["g_optimizer"])
        self.logger.debug(f"load checkpoint: g_optimizer")

        if new_g_lr is not None:
            self.update_learning_rate(self.g_optimizer, new_g_lr)
            self.logger.info(f"g_optimizer: lr is changed to {new_g_lr}")

        if load_scheduler:
            self.g_scheduler.load_state_dict(training_state_ckpt["g_scheduler"])
            self.logger.debug(f"load checkpoint: g_scheduler")
        else:
            self.logger.warn("g_scheduler is not loaded")

        if self.aux_optimizer:
            self.aux_optimizer.load_state_dict(
                training_state_ckpt["aux_optimizer"]
            )
            self.logger.debug(f"load checkpoint: aux_optimizer")
