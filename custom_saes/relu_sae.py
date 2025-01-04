import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download
import numpy as np
import json
from typing import Optional

import custom_saes.base_sae as base_sae


class ReluSAE(base_sae.BaseSAE):
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

    def encode(self, x: torch.Tensor):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon

    @torch.no_grad()
    def normalize_decoder(self):
        """
        This is useful for doing analysis where e.g. feature activation magnitudes are important
        If training the SAE using the Anthropic April update, the decoder weights are not normalized
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        print("Decoder vectors are not normalized. Normalizing.")

        test_input = torch.randn(10, self.cfg.d_in).to(dtype=self.dtype, device=self.device)
        initial_output = self(test_input)

        self.W_dec.data /= norms[:, None]

        new_norms = torch.norm(self.W_dec, dim=1)
        assert torch.allclose(new_norms, torch.ones_like(new_norms))

        self.W_enc *= norms
        self.b_enc *= norms

        new_output = self(test_input)

        max_diff = torch.abs(initial_output - new_output).max()
        print(f"Max difference in output: {max_diff}")

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert torch.allclose(initial_output, new_output, atol=1e-4)


def load_dictionary_learning_relu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> ReluSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config, "r") as f:
        config = json.load(f)

    assert layer == config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = ReluSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "StandardTrainer":
        sae.cfg.architecture = "standard"
    elif config["trainer"]["trainer_class"] == "PAnnealTrainer":
        sae.cfg.architecture = "p_anneal"
    elif config["trainer"]["trainer_class"] == "StandardTrainerAprilUpdate":
        sae.cfg.architecture = "standard_april_update"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        sae.normalize_decoder()

    return sae


if __name__ == "__main__":
    repo_id = "canrager/lm_sae"
    filename = "pythia70m_sweep_standard_ctx128_0712/resid_post_layer_4/trainer_8/ae.pt"
    layer = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "EleutherAI/pythia-70m-deduped"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_relu_sae(repo_id, filename, layer, model_name, device, dtype)

    model = HookedTransformer.from_pretrained(model_name, device=device)

    test_input = "The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science"

    _, cache = model.run_with_cache(test_input, prepend_bos=True)
    acts = cache[hook_name]

    encoded_acts = sae.encode(acts)
    decoded_acts = sae.decode(encoded_acts)

    l0 = (encoded_acts[:, 1:] > 0).float().sum(-1).detach()
    print(f"average l0: {l0.mean().item()}")
