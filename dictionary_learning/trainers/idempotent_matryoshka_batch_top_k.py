import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from collections import namedtuple
from typing import Optional
from math import isclose

from dictionary_learning.dictionary_learning.trainers.idempotent import iou
from dictionary_learning.dictionary_learning.trainers.jumprelu import StepFunction

from ..trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE
from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


class IdempotentMatryoshkaBatchTopKTrainer(MatryoshkaBatchTopKTrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        group_fractions: list[float],
        group_weights: Optional[list[float]] = None,
        dict_class: type = MatryoshkaBatchTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        idempotency_penalty: float = 1.0,
        bandwidth: float = 0.001,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        k_anneal_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "IdempotentMatryoshkaBatchTopKSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(
            steps, 
            activation_dim, 
            dict_size, 
            k, 
            layer, 
            lm_name, 
            group_fractions, 
            group_weights, 
            dict_class, 
            lr, 
            auxk_alpha, 
            warmup_steps, 
            decay_start, 
            threshold_beta, 
            threshold_start_step, 
            k_anneal_steps, 
            seed, 
            device, 
            wandb_name, 
            submodule_name
        )
        self.idempotency_coefficient = idempotency_penalty
        self.bandwidth = bandwidth

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_reconstruct = t.zeros_like(x) + self.ae.b_dec
        total_l2_loss = 0.0
        l2_losses = t.tensor([]).to(self.device)

        # We could potentially refactor the ae class to use W_dec_chunks instead of W_dec, may be more efficient
        W_dec_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)

        for i in range(self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            l2_loss = (x - x_reconstruct).pow(2).sum(
                dim=-1
            ).mean() * self.group_weights[i]
            total_l2_loss += l2_loss
            l2_losses = t.cat([l2_losses, l2_loss.unsqueeze(0)])

        min_l2_loss = l2_losses.min().item()
        max_l2_loss = l2_losses.max().item()
        mean_l2_loss = l2_losses.mean()

        self.effective_l0 = self.k

        f_reconstruct, _, _ = self.ae.encode(
            x_reconstruct, return_active=True, use_threshold=False
        )

        active_features = F.normalize(f, dim=-1)
        active_features_recon = F.normalize(f_reconstruct, dim=-1)
        intersection = (active_features * active_features_recon).sum(dim=-1).mean()
        idempotency_loss = (
            - self.idempotency_coefficient * intersection / self.effective_l0
        )

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        auxk_loss = self.get_auxiliary_loss(
            (x - x_reconstruct).detach(), post_relu_acts_BF
        )
        loss = mean_l2_loss + self.auxk_alpha * auxk_loss + idempotency_loss

        if not logging:
            return loss
        else:
            iou_score = iou(active_features > 0, active_features_recon > 0)
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_reconstruct,
                f,
                {
                    "l2_loss": mean_l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "min_l2_loss": min_l2_loss,
                    "max_l2_loss": max_l2_loss,
                    "idempotency_loss": idempotency_loss.item(),
                    "iou": iou_score,
                },
            )

    @property
    def config(self):
        return {
            "trainer_class": "MatryoshkaBatchTopKTrainer",
            "dict_class": "MatryoshkaBatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "idempotency_penalty": self.idempotency_coefficient,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "group_fractions": self.group_fractions,
            "group_weights": self.group_weights,
            "group_sizes": self.group_sizes,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
