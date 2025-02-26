import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import sae_lens

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


def get_layer_activations(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: BatchEncoding,
    source_pos_B: torch.Tensor,
) -> torch.Tensor:
    acts_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal acts_BLD
        acts_BLD = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(
        gather_target_act_hook
    )

    _ = model(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs.get("attention_mask", None),
    )

    handle.remove()

    assert acts_BLD is not None

    acts_BD = acts_BLD[list(range(acts_BLD.shape[0])), source_pos_B, :]

    return acts_BD


def apply_transformation_matrix(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: BatchEncoding,
    source_rep_BD: torch.Tensor,
    transform_matrix_DD: torch.Tensor,  # Square matrix for transformation
    sae: sae_lens.SAE,  # Kept for API compatibility
    temperature: float,  # Kept for API compatibility
    base_pos_B: torch.Tensor,
    training_mode: bool = False,  # Kept for API compatibility
) -> torch.Tensor:
    acts_BLD = None

    def intervention_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            resid_BLD = outputs[0]
            rest = outputs[1:]
        else:
            raise ValueError("Unexpected output shape")

        # Get the base activations at the target position
        resid_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]

        # Apply the transformation matrix directly to the source representation
        # This gives us much more flexibility than the binary mask
        modified_resid_BD = torch.matmul(source_rep_BD, transform_matrix_DD)

        # Replace the base activations with the transformed source activations
        resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = modified_resid_BD

        return (resid_BLD, *rest)

    handle = model.model.layers[target_layer].register_forward_hook(intervention_hook)

    outputs = model(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs.get("attention_mask", None),
    )

    handle.remove()

    return outputs.logits


class SupervisedBaseline(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: RAVELEvalConfig,
        sae: sae_lens.SAE,  # Kept for API compatibility
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae  # Kept for API compatibility
        self.layer_intervened = sae.cfg.hook_layer

        # Get the hidden dimension from the model
        hidden_dim = model.config.hidden_size

        # Initialize a square transformation matrix
        # Using Xavier/Glorot initialization for better training dynamics
        # self.transform_matrix = torch.nn.Parameter(
        #     torch.nn.init.xavier_uniform_(
        #         torch.zeros(
        #             hidden_dim, hidden_dim, device=model.device, dtype=model.dtype
        #         )
        #     ),
        #     requires_grad=True,
        # )

        # Add identity initialization option (can be uncommented if needed)
        self.binary_mask = torch.nn.Parameter(
            torch.eye(hidden_dim, device=model.device, dtype=model.dtype),
            requires_grad=True,
        )

        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature = 1e-2  # Kept for API compatibility

    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,  # Kept for API compatibility
    ):
        with torch.no_grad():
            # Get source representation
            source_rep = get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL, source_pos_B
            )

        logits = apply_transformation_matrix(
            self.model,
            self.layer_intervened,
            base_encoding_BL,
            source_rep,
            self.binary_mask,
            self.sae,  # Passed for API compatibility
            self.temperature,  # Passed for API compatibility
            base_pos_B,
            training_mode,  # Passed for API compatibility
        )

        predicted = logits.argmax(dim=-1)

        # Format outputs
        predicted_text = []
        for i in range(logits.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted[i]).split()[-1])

        return logits, predicted_text
