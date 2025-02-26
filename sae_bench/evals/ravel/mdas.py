import torch
import torch.nn as nn
import torch.nn.functional as F
import sae_lens
from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sae_bench.evals.ravel.mdbm import (
    get_layer_activations,
    compute_loss,
    get_cause_isolation_scores,
)
from sae_bench.evals.ravel.das_config import DASConfig
from sae_bench.evals.ravel.intervention import get_prompt_pairs


class MDAS(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        config: DASConfig,
    ):
        super(MDAS, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.layer_intervened = int(config.layer_intervened)
        self.d_subspace = config.d_subspace
        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.dtype = model.dtype

        # Create a custom module to hold the rotation matrix
        self.rotation_matrix = nn.Parameter(
            torch.eye(model.config.hidden_size, device=model.device, dtype=model.dtype)
        )

    def patch_subspace(self, base_encoding_BL, base_pos_B, source_rep_BD):
        def intervention_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                resid_BLD = outputs[0]
                rest = outputs[1:]
            else:
                raise ValueError("Unexpected output shape")

            # Compute activations in rotated space
            base_rep_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]
            base_rotated_BD = torch.matmul(base_rep_BD, self.rotation_matrix)
            source_rotated_BD = torch.matmul(source_rep_BD, self.rotation_matrix)

            # Patch source -> base in rotated, smaller subspace
            base_rotated_BD[:, : self.d_subspace] = source_rotated_BD[:, : self.d_subspace]
            base_rep_BD = torch.matmul(base_rotated_BD, self.rotation_matrix.T)
            resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = base_rep_BD

            return (resid_BLD, *rest)

        handle = self.model.model.layers[self.layer_intervened].register_forward_hook(
            intervention_hook
        )

        outputs = self.model(
            input_ids=base_encoding_BL["input_ids"].to(self.device),
            attention_mask=base_encoding_BL.get("attention_mask", None),
        )

        handle.remove()

        return outputs.logits

    def forward(self, base_encoding_BL, source_encoding_BL, base_pos_B, source_pos_B):
        source_rep_BD = get_layer_activations(
            self.model, self.layer_intervened, source_encoding_BL, source_pos_B
        )
        logits = self.patch_subspace(base_encoding_BL, base_pos_B, source_rep_BD)

        # Greedy decode logits
        predicted = logits.argmax(dim=-1)
        predicted_text = []
        for i in range(logits.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted[i]).split()[-1])

        return logits, predicted_text

    def orthogonalize_(self):
        with torch.no_grad():
            w = self.rotation_matrix.to(torch.float32)
            U, _, V = torch.svd(w)
            orthogonal_w = U @ V.T
            self.rotation_matrix.copy_(orthogonal_w.to(self.dtype))
        return self.rotation_matrix


def get_validation_loss(mdas: MDAS, val_loader: torch.utils.data.DataLoader):
    """Compute validation loss across the validation dataset"""
    mdas.eval()
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
            intervened_logits_BLV, _ = mdas(
                base_encodings_BL, source_encodings_BL, base_pos_B, source_pos_B
            )
            loss, accuracy = compute_loss(intervened_logits_BLV, source_pred_B)

            # Calculate cause and isolation scores
            cause_score, isolation_score, cause_count, isolation_count = get_cause_isolation_scores(
                intervened_logits_BLV, source_pred_B, base_pred_B
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
    avg_val_cause_score = val_cause_score / total_cause_count if total_cause_count > 0 else 0
    avg_val_isolation_score = (
        val_isolation_score / total_isolation_count if total_isolation_count > 0 else 0
    )

    return avg_val_loss, avg_val_accuracy, avg_val_cause_score, avg_val_isolation_score


def train_mdas(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast,
    config: DASConfig,
    train_loader,
    val_loader,
    verbose: bool = False,
) -> dict[str, float]:

    mdas = MDAS(
        model,
        tokenizer,
        config,
    ).to(model.device)

    optimizer = torch.optim.Adam([mdas.rotation_matrix], lr=config.learning_rate)

    # Get initial validation loss
    (
        initial_val_loss,
        initial_val_accuracy,
        initial_val_cause_score,
        initial_val_isolation_score,
    ) = get_validation_loss(mdas, val_loader)
    if verbose:
        print(
            f"Initial validation loss: {initial_val_loss:.4f}, accuracy: {initial_val_accuracy:.4f}, cause score: {initial_val_cause_score:.4f}, isolation score: {initial_val_isolation_score:.4f}"
        )

    best_val_loss = initial_val_loss
    patience_counter = 0

    for epoch in range(config.num_epochs):
        mdas.train()
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

            intervened_logits_BLV, _ = mdas(
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
            )
            loss, accuracy = compute_loss(intervened_logits_BLV, source_pred_B)

            loss.backward()
            optimizer.step()
            mdas.orthogonalize_()

            train_loss += loss.item()
            train_accuracy += accuracy.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_train_accuracy = train_accuracy / batch_count if batch_count > 0 else 0
        # Validation

        # Print losses if verbose
        (
            avg_val_loss,
            avg_val_accuracy,
            avg_val_cause_score,
            avg_val_isolation_score,
        ) = get_validation_loss(mdas, val_loader)
        if verbose:
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {avg_train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {avg_val_accuracy:.4f}, "
                f"Val Cause Score: {avg_val_cause_score:.4f}, "
                f"Val Isolation Score: {avg_val_isolation_score:.4f}"
            )

        # Early stopping
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     if verbose:
        #         print(f"  New best validation loss: {best_val_loss:.4f}")
        # else:
        #     patience_counter += 1
        #     if verbose:
        #         print(f"  No improvement for {patience_counter} epochs")

        # if patience_counter >= config.early_stop_patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    (
        final_val_loss,
        final_val_accuracy,
        final_val_cause_score,
        final_val_isolation_score,
    ) = get_validation_loss(mdas, val_loader)

    disentangle_score = (final_val_cause_score + final_val_isolation_score) / 2

    results = {
        "val_loss": final_val_loss,
        "val_accuracy": final_val_accuracy,
        "disentangle_score": disentangle_score,
        "cause_score": final_val_cause_score,
        "isolation_score": final_val_isolation_score,
    }

    return results


if __name__ == "__main__":
    import os
    import random
    from sae_bench.evals.ravel.main import create_dataloaders
    from sae_bench.evals.ravel.instance import create_filtered_dataset
    import sae_bench.sae_bench_utils.general_utils as general_utils

    # Initialize experiment parameters
    config = DASConfig()
    device = "cuda:0"
    config.model_dir = "/share/u/models"
    entity_class = list(config.entity_attribute_selection.keys())[0]
    attributes = config.entity_attribute_selection[entity_class]
    cause_attribute = attributes[0]
    iso_attributes = attributes[1:]

    # Initialize tokenizer
    print("Loading model")
    LLM_NAME_MAP = {"gemma-2-2b": "google/gemma-2-2b"}
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    config.model_name = LLM_NAME_MAP[config.model_name]

    if "gemma" in config.model_name:
        model_kwargs = {"attn_implementation": "eager"}
    else:
        model_kwargs = {}

    model_cache_folder = os.path.join(config.model_dir, f'models--{config.model_name.replace("/", "--")}')
    is_existing_path = os.path.exists(model_cache_folder)
    
    if is_existing_path:
        print(f"Model cache folder already exists at {model_cache_folder}")
    else:    
        print(f"Model cache folder does not exist at {model_cache_folder}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        cache_dir=config.model_dir,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Initialize dataset
    print("Creating dataset")
    dataset = create_filtered_dataset(
        model_id=config.model_name,
        chosen_entity=entity_class,
        model=model,
        force_recompute=config.force_dataset_recompute,
        n_samples_per_attribute_class=config.n_samples_per_attribute_class,
        top_n_entities=config.top_n_entities,
        top_n_templates=config.top_n_templates,
        artifact_dir=config.artifact_dir,
        full_dataset_downsample=config.full_dataset_downsample,
    )

    cause_base_prompts, cause_source_prompts = get_prompt_pairs(
        dataset=dataset,
        base_attribute=cause_attribute,
        source_attribute=cause_attribute,
        n_interventions=config.num_pairs_per_attribute,
    )

    cause_base_prompts, cause_source_prompts = get_prompt_pairs(
        dataset=dataset,
        base_attribute=cause_attribute,
        source_attribute=cause_attribute,
        n_interventions=config.num_pairs_per_attribute,
    )

    iso_base_prompts = []
    iso_source_prompts = []
    for iso_attr in iso_attributes:
        attr_base_prompts, attr_source_prompts = get_prompt_pairs(
            dataset=dataset,
            base_attribute=iso_attr,
            source_attribute=iso_attr,
            n_interventions=config.num_pairs_per_attribute,
        )
        iso_base_prompts.extend(attr_base_prompts)
        iso_source_prompts.extend(attr_source_prompts)

    combined = list(zip(iso_base_prompts, iso_source_prompts))
    random.shuffle(combined)
    iso_base_prompts, iso_source_prompts = zip(*combined)

    # Truncate to match the length of cause prompts
    cause_length = len(cause_base_prompts)
    iso_base_prompts = list(iso_base_prompts[:cause_length])
    iso_source_prompts = list(iso_source_prompts[:cause_length])

    print(
        f"Using {len(cause_base_prompts)} cause prompt pairs and {len(iso_base_prompts)} ISO prompt pairs"
    )

    train_loader, val_loader = create_dataloaders(
        cause_base_prompts,
        cause_source_prompts,
        iso_base_prompts,
        iso_source_prompts,
        model,
        config,
        train_test_split=config.train_test_split,
    )

    print("Training MDAS")
    train_mdas(
        model,
        tokenizer,
        config,
        train_loader,
        val_loader,
        verbose=True,
    )
