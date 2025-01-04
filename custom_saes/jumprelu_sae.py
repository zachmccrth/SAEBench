import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Optional

import custom_saes.custom_sae_config as sae_config
import custom_saes.base_sae as base_sae


class JumpReLUSAE(base_sae.BaseSAE):
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
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.threshold = nn.Parameter(torch.zeros(d_sae)).to(dtype=dtype, device=device)

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReLUSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cpu() for k, v in params.items()}

    d_in = params["W_enc"].shape[0]
    d_sae = params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReLUSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    sae.cfg.architecture = "jumprelu"

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder norms are not normalized. Implement a normalization method.")

    return sae


if __name__ == "__main__":
    layer = 20

    repo_id = "google/gemma-scope-2b-pt-res"
    filename = f"layer_{layer}/width_16k/average_l0_71/params.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model_name = "google/gemma-2-2b"

    sae = load_gemma_scope_jumprelu_sae(repo_id, filename, layer, model_name, device, dtype)
