import torch
import torch.nn as nn
import torch.nn.functional as F
import sae_lens
from transformers import AutoModelForCausalLM, BatchEncoding, AutoTokenizer

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.supervised_baseline import SupervisedBaseline


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


def apply_binary_mask(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: BatchEncoding,
    source_rep_BD: torch.Tensor,
    binary_mask_F: torch.Tensor,
    sae: sae_lens.SAE,
    temperature: float,
    base_pos_B: torch.Tensor,
    training_mode: bool = False,
) -> torch.Tensor:
    acts_BLD = None

    def intervention_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            resid_BLD = outputs[0]
            rest = outputs[1:]
        else:
            raise ValueError("Unexpected output shape")

        source_act_BF = sae.encode(source_rep_BD)
        resid_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]
        base_act_BF = sae.encode(resid_BD)

        # Use true binary mask in eval mode, sigmoid in training mode
        if not training_mode:
            mask_values_F = (binary_mask_F > 0).to(dtype=binary_mask_F.dtype)
        else:
            mask_values_F = torch.sigmoid(binary_mask_F / temperature)

        modified_act_BF = (
            1 - mask_values_F
        ) * base_act_BF + mask_values_F * source_act_BF
        modified_resid_BD = sae.decode(modified_act_BF)

        resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = modified_resid_BD

        return (resid_BLD, *rest)

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
        self.layer_intervened = sae.cfg.hook_layer
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=model.device, dtype=model.dtype),
            requires_grad=True,
        )
        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature: float = 1e-2

    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,
    ):
        with torch.no_grad():
            # Get source representation
            source_rep = get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL, source_pos_B
            )

        logits = apply_binary_mask(
            self.model,
            self.layer_intervened,
            base_encoding_BL,
            source_rep,
            self.binary_mask,
            self.sae,
            self.temperature,
            base_pos_B,
            training_mode,
        )

        predicted = logits.argmax(dim=-1)

        # Format outputs
        predicted_text = []
        for i in range(logits.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted[i]).split()[-1])

        return logits, predicted_text


def compute_loss(intervened_logits_BLV, target_attr_B):
    """
    Compute multi-task loss combining:
    - Cause loss: Target attribute should match source
    - Iso loss: Other attributes should match base

    Returns:
        Tuple of (loss, accuracy) where accuracy is the raw prediction accuracy
        for the final token
    """
    cause_loss = F.cross_entropy(intervened_logits_BLV[:, -1, :], target_attr_B)

    # Calculate accuracy
    predictions = intervened_logits_BLV[:, -1, :].argmax(dim=-1)
    accuracy = (predictions == target_attr_B).float().mean()

    return cause_loss, accuracy

    # iso_losses = []
    # for attr in other_attrs:
    #     iso_losses.append(F.cross_entropy(base_outputs, attr))
    # iso_loss = torch.stack(iso_losses).mean()

    # return cause_loss + iso_loss


def get_cause_isolation_scores(intervened_logits_BLV, source_pred_B, base_pred_B):
    """
    Calculate cause and isolation scores based on predictions.

    Args:
        intervened_logits_BLV: Logits from the intervened model
        source_pred_B: Target predictions from source examples
        base_pred_B: Target predictions from base examples

    Returns:
        Tuple of (cause_score, isolation_score, cause_count, isolation_count)
    """
    predictions = intervened_logits_BLV[:, -1, :].argmax(dim=-1)

    # Identify cause and isolation examples
    is_isolation = base_pred_B == source_pred_B
    is_cause = ~is_isolation

    # Count examples in each category
    cause_count = is_cause.sum().item()
    isolation_count = is_isolation.sum().item()

    # Calculate accuracy for each category
    cause_correct = ((predictions == source_pred_B) & is_cause).sum().item()
    isolation_correct = ((predictions == base_pred_B) & is_isolation).sum().item()

    # Calculate scores (handle division by zero)
    cause_score = cause_correct / cause_count if cause_count > 0 else 0.0
    isolation_score = (
        isolation_correct / isolation_count if isolation_count > 0 else 0.0
    )

    return cause_score, isolation_score, cause_count, isolation_count


def get_validation_loss(mdbm: MDBM, val_loader: torch.utils.data.DataLoader):
    """Compute validation loss across the validation dataset"""
    mdbm.eval()
    val_loss = 0
    val_accuracy = 0
    val_batch_count = 0
    val_cause_score = 0
    val_isolation_score = 0
    total_cause_count = 0
    total_isolation_count = 0

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
                base_encodings_BL, source_encodings_BL, base_pos_B, source_pos_B
            )
            loss, accuracy = compute_loss(intervened_logits_BLV, source_pred_B)

            # Calculate cause and isolation scores
            cause_score, isolation_score, cause_count, isolation_count = (
                get_cause_isolation_scores(
                    intervened_logits_BLV, source_pred_B, base_pred_B
                )
            )

            val_loss += loss.item()
            val_accuracy += accuracy.item()
            val_cause_score += cause_score * cause_count
            val_isolation_score += isolation_score * isolation_count
            total_cause_count += cause_count
            total_isolation_count += isolation_count
            val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
    avg_val_accuracy = val_accuracy / val_batch_count if val_batch_count > 0 else 0
    avg_val_cause_score = (
        val_cause_score / total_cause_count if total_cause_count > 0 else 0
    )
    avg_val_isolation_score = (
        val_isolation_score / total_isolation_count if total_isolation_count > 0 else 0
    )

    return avg_val_loss, avg_val_accuracy, avg_val_cause_score, avg_val_isolation_score


def train_mdbm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: RAVELEvalConfig,
    sae: sae_lens.SAE,
    train_loader,
    val_loader,
    verbose: bool = False,
):
    initial_temperature = 1e-2
    final_temperature = 1e-7
    temperature_schedule = torch.logspace(
        torch.log10(torch.tensor(initial_temperature)),
        torch.log10(torch.tensor(final_temperature)),
        config.num_epochs,
        device=model.device,
        dtype=model.dtype,
    )

    mdbm = MDBM(
        model,
        tokenizer,
        config,
        sae,
    ).to(model.device)

    # mdbm = SupervisedBaseline(
    #     model,
    #     tokenizer,
    #     config,
    #     sae,
    # ).to(model.device)

    optimizer = torch.optim.Adam([mdbm.binary_mask], lr=config.learning_rate)

    # Get initial validation loss
    (
        initial_val_loss,
        initial_val_accuracy,
        initial_val_cause_score,
        initial_val_isolation_score,
    ) = get_validation_loss(mdbm, val_loader)
    if verbose:
        print(
            f"Initial validation loss: {initial_val_loss:.4f}, accuracy: {initial_val_accuracy:.4f}, cause score: {initial_val_cause_score:.4f}, isolation score: {initial_val_isolation_score:.4f}"
        )

    best_val_loss = initial_val_loss
    patience_counter = 0

    for epoch in range(config.num_epochs):
        mdbm.temperature = temperature_schedule[epoch].item()
        mdbm.train()
        train_loss = 0
        train_accuracy = 0
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
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                training_mode=True,
            )
            loss, accuracy = compute_loss(
                intervened_logits_BLV, source_pred_B
            )  # TODO: only caus score currently used, add iso score

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_train_accuracy = train_accuracy / batch_count if batch_count > 0 else 0
        # Validation
        avg_val_loss, avg_val_accuracy, avg_val_cause_score, avg_val_isolation_score = (
            get_validation_loss(mdbm, val_loader)
        )

        # Print losses if verbose
        if verbose:
            percent_above_zero = (mdbm.binary_mask > 0).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {avg_train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {avg_val_accuracy:.4f}, "
                f"Percent above zero: {percent_above_zero:.4f}, "
                f"Val Cause Score: {avg_val_cause_score:.4f}, "
                f"Val Isolation Score: {avg_val_isolation_score:.4f}"
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

        # if patience_counter >= config.early_stop_patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    return mdbm
