import torch
import torch.nn as nn
from typing import Optional
import custom_saes.custom_sae_config as sae_config


class BaseSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: Optional[str] = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        # b_enc and b_dec don't have to be used in the encode/decode methods
        # if your SAE doesn't use biases, leave them as zeros
        # NOTE: core() checks for cosine similarity with b_enc, so it's nice to have the field available
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.cfg = sae_config.CustomSAEConfig(
            model_name, d_in=d_in, d_sae=d_sae, hook_name=hook_name, hook_layer=hook_layer
        )
        self.cfg.dtype = self.dtype.__str__().split(".")[1]

    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        if torch.allclose(norms, torch.ones_like(norms)):
            return True
        else:
            return False

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self
