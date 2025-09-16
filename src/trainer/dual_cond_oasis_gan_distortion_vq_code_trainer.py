from typing import Dict, Tuple, Optional, Union, List

import torch
from torch import Tensor

from src.utils.registry import TRAINER_REGISTRY

from .dual_cond_gan_distortion_vq_code_trainer import (
    DualBetaCondGanDistortionVqCodeTrainer,
)
from .dual_cond_rate_distortion_vq_code_trainer import (
    DualBetaModelOutput,
)


@TRAINER_REGISTRY.register()
class DualBetaCondOasisGanDistortionVqFusionTrainer(
    DualBetaCondGanDistortionVqCodeTrainer
):

    def calc_g_loss(
        self,
        outputs: DualBetaModelOutput,
    ) -> Tuple[Tensor, Dict]:
        real_images = outputs.real_images
        fake_images = outputs.fake_images
        gt_vq_latent = outputs.gt_vq_latent
        out_vq_latent = outputs.out_vq_latent
        gt_vq_indices = outputs.gt_vq_indices
        out_vq_logits = outputs.out_vq_logits
        beta_rate = outputs.beta_rate
        beta_vq = outputs.beta_vq
        y_hat = outputs.y_hat
        other_outputs = outputs.other_outputs

        log_dict = {}
        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        ## Image Loss
        distortion = self.distortion_loss(real_images, fake_images, **other_outputs)
        log_dict["distortion"] = distortion
        loss_total = distortion

        perceptual = self.perceptual_loss(real_images, fake_images)
        log_dict["perceptual"] = perceptual
        loss_total += perceptual

        # **** OASIS Adv Loss ****************
        g_fake = self.discriminator(
            fake_images, beta_1=beta_rate, beta_2=beta_vq, y_hat=y_hat, **other_outputs
        )
        adv_loss = self.gan_loss(g_fake, gt_vq_indices, is_real=True, is_disc=False)
        log_dict["adv"] = adv_loss
        loss_total += adv_loss

        ## Code Loss
        code_distortion_loss = getattr(self, "code_distortion_loss")
        if code_distortion_loss is not None:
            code_distortion = code_distortion_loss(
                gt_vq_latent, out_vq_latent, **other_outputs
            )
            log_dict["code_distortion"] = code_distortion
            loss_total += code_distortion

        code_gram_loss = getattr(self, "code_gram_loss")
        if code_gram_loss is not None:
            code_gram = code_gram_loss(gt_vq_latent, out_vq_latent, **other_outputs)
            log_dict["code_gram"] = code_gram
            loss_total += code_gram

        code_ce_loss = getattr(self, "code_ce_loss")
        if code_ce_loss is not None:
            code_ce = code_ce_loss(out_vq_logits, gt_vq_indices)
            log_dict["code_ce"] = code_ce
            loss_total += code_ce

        return loss_total, log_dict

    def calc_d_loss(
        self,
        d_real: Tensor,
        d_fake: Tensor,
        gt_vq_indices: Tensor,
        disc_gt_vq_indices: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        if self.mc_sampling:
            l_d_real = self.gan_loss(d_real, disc_gt_vq_indices, is_real=True, is_disc=True)
        else:
            l_d_real = self.gan_loss(d_real, gt_vq_indices, is_real=True, is_disc=True)
        l_d_real = l_d_real * 0.5

        l_d_fake = self.gan_loss(d_fake, gt_vq_indices, is_real=False, is_disc=True)
        l_d_fake = l_d_fake * 0.5

        d_loss = l_d_real + l_d_fake

        out_d_real, out_d_fake = self.calc_avg_d_score_for_log(d_real, d_fake)
        log_dict = {
            "d_real": l_d_real,
            "d_fake": l_d_fake,
            "d_total": d_loss,
            "out_d_real": out_d_real,
            "out_d_fake": out_d_fake,
        }
        return d_loss, log_dict

    @staticmethod
    def calc_avg_d_score_for_log(d_real, d_fake):
        out_d_real = torch.mean(d_real[:, 1:, :, :].detach())
        out_d_fake = torch.mean(d_fake[:, 1:, :, :].detach())
        return out_d_real, out_d_fake
