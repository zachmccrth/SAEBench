import torch
import torch.nn as nn
import torch.nn.functional as F
import sae_lens
from transformers import AutoModelForCausalLM, BatchEncoding, AutoTokenizer

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


def get_layer_activations(
    model: AutoModelForCausalLM, target_layer: int, inputs: BatchEncoding
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

    return acts_BLD


def apply_binary_mask(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: BatchEncoding,
    source_rep_BLD: torch.Tensor,
    binary_mask_F: torch.Tensor,
    sae: sae_lens.SAE,
    temperature: float,
) -> torch.Tensor:
    acts_BLD = None

    def intervention_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            resid_BLD = outputs[0]
            rest = outputs[1:]
        else:
            raise ValueError("Unexpected output shape")

        source_act_BLF = sae.encode(source_rep_BLD)
        base_act_BLF = sae.encode(resid_BLD)
        mask_values_F = torch.sigmoid(binary_mask_F / temperature)

        modified_act_BLF = (
            1 - mask_values_F
        ) * base_act_BLF + mask_values_F * source_act_BLF
        modified_resid_BLD = sae.decode(modified_act_BLF)

        return (modified_resid_BLD, *rest)

    handle = model.model.layers[target_layer].register_forward_hook(intervention_hook)

    outputs = model(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs.get("attention_mask", None),
    )

    handle.remove()

    return outputs.logits


class MDBM(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: RAVELEvalConfig,
        sae: sae_lens.SAE,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer_intervened = torch.tensor(
            sae.cfg.hook_layer, dtype=torch.int32, device=model.device
        )
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=model.device, dtype=model.dtype),
            requires_grad=True,
        )
        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature = 1e-2

    def forward(self, base_encoding_BL, source_encoding_BL, base_pos_B, source_pos_B):
        with torch.no_grad():
            # Get source representation
            source_rep = get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL
            )

        logits = apply_binary_mask(
            self.model,
            self.layer_intervened,
            base_encoding_BL,
            source_rep,
            self.binary_mask,
            self.sae,
            self.temperature,
        )

        predicted = logits.argmax(dim=-1)

        # Format outputs
        predicted_text = []
        for i in range(logits.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted[i]).split()[-1])

        return logits, predicted_text

    def compute_loss(self, intervened_logits_BLV, target_attr_B):
        """
        Compute multi-task loss combining:
        - Cause loss: Target attribute should match source
        - Iso loss: Other attributes should match base
        """
        cause_loss = F.cross_entropy(intervened_logits_BLV[:, -1, :], target_attr_B)
        return cause_loss

        # iso_losses = []
        # for attr in other_attrs:
        #     iso_losses.append(F.cross_entropy(base_outputs, attr))
        # iso_loss = torch.stack(iso_losses).mean()

        # return cause_loss + iso_loss


def train_mdbm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: RAVELEvalConfig,
    sae: sae_lens.SAE,
    train_loader,
    val_loader,
    verbose: bool = False,
):
    mdbm = MDBM(
        model,
        tokenizer,
        config,
        sae,
    ).to(model.device)
    optimizer = torch.optim.Adam(mdbm.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        mdbm.train()
        train_loss = 0
        batch_count = 0

        for batch in train_loader:
            (
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
            ) = batch

            optimizer.zero_grad()

            intervened_logits_BLV, _ = mdbm(
                source_encodings_BL, base_encodings_BL, base_pos_B, source_pos_B
            )
            loss = mdbm.compute_loss(
                intervened_logits_BLV, base_pred_B
            )  # TODO: only caus score currently used, add iso score

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0

        # Validation
        mdbm.eval()
        val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                (
                    base_encodings_BL,
                    source_encodings_BL,
                    base_pos_B,
                    source_pos_B,
                    base_pred_B,
                    source_pred_B,
                ) = batch
                intervened_logits_BLV, _ = mdbm(
                    source_encodings_BL, base_encodings_BL, base_pos_B, source_pos_B
                )
                val_loss += mdbm.compute_loss(intervened_logits_BLV, base_pred_B).item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0

        # Print losses if verbose
        if verbose:
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if verbose:
                print(f"  New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if verbose:
                print(f"  No improvement for {patience_counter} epochs")

        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    return mdbm
