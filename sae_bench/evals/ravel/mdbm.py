import torch
import torch.nn as nn
import torch.nn.functional as F
import sae_lens

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


class MDBM(nn.Module):
    def __init__(self, model, config: RAVELEvalConfig, sae: sae_lens.SAE):
        super().__init__()

        self.model = model
        self.sae = sae
        self.layer_intervened = torch.tensor(
            config.layer_intervened, dtype=torch.int32, device=model.device
        )
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=model.device, dtype=model.dtype),
            requires_grad=True,
        )
        self.batch_size = config.batch_size
        self.device = model.device

    def forward(self, base_encoding_BL, source_encoding_BL, base_pos_B, source_pos_B):
        with self.model.trace() as tracer:
            # Get source representation
            with tracer.invoke(source_encoding_BL) as runner:
                source_rep = self.model.model.layers[self.layer_intervened].output[0]

            print(source_rep.shape)
            raise ValueError("Stop here")

            # Get base representation
            with tracer.invoke(base_encoding_BL) as runner:
                base_rep = (
                    self.model.model.layers[self.layer_intervened].output[0].clone()
                )

                # Apply rotation using the weight from rotation module
                source_rotated = torch.matmul(source_rep, self.rotation_module.weight)
                base_rotated = torch.matmul(base_rep, self.rotation_module.weight)

                # Intervention: Replace target dimensions
                base_rotated[range(len(base_pos_B)), base_pos_B, : self.d_subspace] = (
                    source_rotated[
                        range(len(source_pos_B)), source_pos_B, : self.d_subspace
                    ]
                )

                # Rotate back using transpose of the weight
                base_intervened = torch.matmul(
                    base_rotated, self.rotation_module.weight.T
                )

                # Update model representation
                self.model.model.layers[self.layer_intervened].output = (
                    base_intervened,
                )

                # Get model outputs
                logits = self.model.lm_head.output
                predicted = logits.argmax(dim=-1)

        # Format outputs
        predicted_text = []
        for i in range(logits.shape[0]):
            predicted_text.append(self.model.tokenizer.decode(predicted[i]).split()[-1])

        return logits, predicted_text

    def compute_loss(self, intervened_logits, target_attr):
        """
        Compute multi-task loss combining:
        - Cause loss: Target attribute should match source
        - Iso loss: Other attributes should match base
        """
        cause_loss = F.cross_entropy(intervened_logits, target_attr)
        return cause_loss

        # iso_losses = []
        # for attr in other_attrs:
        #     iso_losses.append(F.cross_entropy(base_outputs, attr))
        # iso_loss = torch.stack(iso_losses).mean()

        # return cause_loss + iso_loss


def train_mdbm(
    model,
    config: RAVELEvalConfig,
    sae: sae_lens.SAE,
    train_loader,
    val_loader,
):
    mdas = MDBM(
        model,
        config,
        sae,
    ).to(model.device)
    optimizer = torch.optim.Adam(mdas.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        mdas.train()
        train_loss = 0

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

            intervened_logits, _ = mdas(
                source_encodings_BL, base_encodings_BL, base_pos_B, source_pos_B
            )
            loss = mdas.compute_loss(
                intervened_logits, base_pred_B
            )  # TODO: only caus score currently used, add iso score

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        mdas.eval()
        val_loss = 0

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
                intervened_logits, _ = mdas(
                    source_encodings_BL, base_encodings_BL, base_pos_B, source_pos_B
                )
                val_loss += mdas.compute_loss(intervened_logits, base_pred_B)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return mdas
