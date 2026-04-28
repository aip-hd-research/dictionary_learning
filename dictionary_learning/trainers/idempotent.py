from collections import namedtuple

from dictionary_learning.dictionary_learning.trainers.jumprelu import JumpReLUFunction, StepFunction
import torch
import torch.autograd as autograd
from torch import nn
from typing import Optional

from ..dictionary import Dictionary, JumpReluAutoEncoder
from ..trainers.trainer import (
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from ..trainers.jumprelu import JumpReluTrainer

class IdempotentTrainer(JumpReluTrainer):
    """
    Trains an idempotent autoencoder.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=JumpReluAutoEncoder,
        seed: Optional[int] = None,
        # TODO: What's the default lr use in the paper?
        lr: float = 7e-5,
        bandwidth: float = 0.001,
        sparsity_penalty: float = 1.0,
        idempotency_penalty: float = 1.0,
        warmup_steps: int = 1000,  # lr warmup period at start of training and after each resample
        sparsity_warmup_steps: Optional[int] = 2000,  # sparsity warmup period at start of training
        decay_start: Optional[int] = None,  # decay learning rate after this many steps
        target_l0: float = 20.0,
        device: str = "cpu",
        wandb_name: str = "Idempotent",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(
            steps, 
            activation_dim, 
            dict_size, 
            layer, 
            lm_name, 
            dict_class, 
            seed, 
            lr, 
            bandwidth, 
            sparsity_penalty, 
            warmup_steps, 
            sparsity_warmup_steps, 
            decay_start, 
            target_l0, 
            device, 
            wandb_name, 
            submodule_name
        )
        self.idempotency_coefficient = idempotency_penalty

    def loss(self, x: torch.Tensor, step: int, logging=False, **_):
        # Note: We are using threshold, not log_threshold as in this notebook:
        # https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD#scrollTo=yP828a6uIlSO
        # I had poor results when using log_threshold and it would complicate the scale_biases() function

        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.W_enc.dtype)

        pre_jump = x @ self.ae.W_enc + self.ae.b_enc
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)

        active_indices = f.sum(0) > 0
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
        )

        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()

        active_features = StepFunction.apply(f, self.ae.threshold, self.bandwidth)
        l0 = active_features.sum(dim=-1).mean()

        sparsity_loss = (
            self.sparsity_coefficient * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        )

        pre_jump_recon = recon @ self.ae.W_enc + self.ae.b_enc
        f_recon = JumpReLUFunction.apply(pre_jump_recon, self.ae.threshold, self.bandwidth)
        active_features_recon = StepFunction.apply(f_recon, self.ae.threshold, self.bandwidth)

        intersection = (active_features * active_features_recon).sum(dim=-1).mean()
        idempotency_loss = (
            - self.idempotency_coefficient * intersection / self.target_l0
        )

        loss = recon_loss + sparsity_loss + idempotency_loss

        if not logging:
            return loss
        else:
            iou_score = iou(active_features > 0, active_features_recon > 0)
            return namedtuple("LossLog", ["x", "recon", "f", "losses"])(
                x,
                recon,
                f,
                {
                    "l2_loss": recon_loss.item(),
                    "idempotency_loss": idempotency_loss.item(),
                    "iou": iou_score,
                    "loss": loss.item(),
                },
            )

    @property
    def config(self):
        return {
            "trainer_class": "IdempotentTrainer",
            "dict_class": "JumpReluAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "bandwidth": self.bandwidth,
            "sparsity_penalty": self.sparsity_coefficient,
            "idempotency_penalty": self.idempotency_coefficient,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "target_l0": self.target_l0,
        }

def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool() # no-op if already of the correct type
    b = b.bool()
    return ((a * b).sum(dim=-1) / (a + b).sum(dim=-1).clamp_min(1)).mean().detach().cpu().item()