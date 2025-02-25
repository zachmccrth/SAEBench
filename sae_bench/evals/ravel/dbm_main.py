import torch
from transformers import BatchEncoding, AutoTokenizer, AutoModelForCausalLM
import sae_lens

import sae_bench.evals.ravel.mdbm as mdbm
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.instance import create_filtered_dataset
from sae_bench.evals.ravel.intervention import get_prompt_pairs


def inspect_dataloader(
    dataloader, tokenizer, n=2, view_strs: bool = False, max_len: int = 300
):
    """
    Debug function to inspect the first n elements of the dataloader.

    Args:
        dataloader: The dataloader to inspect
        tokenizer: The tokenizer to use for detokenizing
        n: Number of batches to inspect (default: 2)
    """
    print(f"\n{'=' * 50}\nDEBUGGING DATALOADER CONTENTS\n{'=' * 50}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n:
            break

        print(f"\n{'-' * 50}\nBATCH {batch_idx + 1}/{n}\n{'-' * 50}")

        (
            base_encoding_BL,
            source_encoding_BL,
            base_pos_B,
            source_pos_B,
            base_pred_B,
            source_pred_B,
        ) = batch

        # Print shapes
        print("\nSHAPES:")
        print(f"  base_encoding_BL['input_ids']: {base_encoding_BL['input_ids'].shape}")
        print(
            f"  base_encoding_BL['attention_mask']: {base_encoding_BL['attention_mask'].shape}"
        )
        print(
            f"  source_encoding_BL['input_ids']: {source_encoding_BL['input_ids'].shape}"
        )
        print(
            f"  source_encoding_BL['attention_mask']: {source_encoding_BL['attention_mask'].shape}"
        )
        print(f"  base_pos_B: {base_pos_B.shape}")
        print(f"  source_pos_B: {source_pos_B.shape}")
        print(f"  base_pred_B: {base_pred_B.shape}")
        print(f"  source_pred_B: {source_pred_B.shape}")

        if view_strs:
            # Inspect individual examples in the batch
            batch_size = base_encoding_BL["input_ids"].shape[0]
            examples_to_show = min(3, batch_size)  # Show at most 3 examples per batch

            for i in range(examples_to_show):
                print(f"\nEXAMPLE {i + 1}/{examples_to_show}:")

                # Get base sequence
                base_ids = base_encoding_BL["input_ids"][i].tolist()
                base_mask = base_encoding_BL["attention_mask"][i].tolist()
                base_text = tokenizer.decode(base_ids)

                # Get source sequence
                source_ids = source_encoding_BL["input_ids"][i].tolist()
                source_mask = source_encoding_BL["attention_mask"][i].tolist()
                source_text = tokenizer.decode(source_ids)

                # Get positions and predictions
                base_position = base_pos_B[i].item()
                source_position = source_pos_B[i].item()
                base_prediction = base_pred_B[i].item()
                source_prediction = source_pred_B[i].item()

                # Get tokens at the positions of interest
                base_token_at_pos = tokenizer.decode([base_ids[base_position]])
                source_token_at_pos = tokenizer.decode([source_ids[source_position]])
                base_pred_token = tokenizer.decode([base_prediction])
                source_pred_token = tokenizer.decode([source_prediction])

                # Print everything
                print(f"  BASE TEXT: {base_text[:max_len]}... (truncated)")
                print(f"  SOURCE TEXT: {source_text[:max_len]}... (truncated)")
                print(f"  BASE POSITION: {base_position}")
                print(f"  SOURCE POSITION: {source_position}")
                print(f"  TOKEN AT BASE POSITION: '{base_token_at_pos}'")
                print(f"  TOKEN AT SOURCE POSITION: '{source_token_at_pos}'")
                print(f"  BASE PREDICTION: {base_prediction} ('{base_pred_token}')")
                print(
                    f"  SOURCE PREDICTION: {source_prediction} ('{source_pred_token}')"
                )

    print(f"\n{'=' * 50}\nDEBUGGING COMPLETE\n{'=' * 50}")


def main():
    eval_config = RAVELEvalConfig()
    device = "cuda:0"
    # Load model
    if eval_config.model_name == "gemma-2-2b":
        model_id = "google/gemma-2-2b"
        model_kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
    else:
        raise ValueError(f"Invalid model name: {eval_config.model_name}")

    if eval_config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif eval_config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {eval_config.llm_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=llm_dtype,
        **model_kwargs,
    )
    model.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sae_release = "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109"
    sae_id = "blocks.12.hook_resid_post__trainer_5"

    if isinstance(sae_id, str):
        sae = sae_lens.SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device,
        )[0]
    else:
        sae = sae_id
        sae_id = "custom_sae"

    sae = sae.to(device=device, dtype=llm_dtype)

    # Load train/val data
    # dataloader batch contains (base_tokens_BL, source_tokens_BL, base_pos_B, source_pos_B, target_id_B, other_id_B)

    ## Create Ravel dataset
    entity_class = next(iter(eval_config.entity_attribute_selection.keys()))
    print(f"Using entity class: {entity_class}")

    dataset = create_filtered_dataset(
        model_id=model_id,
        chosen_entity=entity_class,
        model=model,
        force_recompute=eval_config.force_dataset_recompute,
        n_samples_per_attribute_class=eval_config.n_samples_per_attribute_class,
        top_n_entities=eval_config.top_n_entities,
        top_n_templates=eval_config.top_n_templates,
        artifact_dir=eval_config.artifact_dir,
        full_dataset_downsample=eval_config.full_dataset_downsample,
    )

    # We are only computing cause
    attribute = "Country"
    other_attribute = "Continent"

    chosen_attributes = eval_config.entity_attribute_selection[entity_class]
    available_attributes = dataset.get_attributes()
    for c in chosen_attributes:
        assert c in available_attributes, f"Attribute {c} not found in dataset"

    print(f"Using attributes: {chosen_attributes}")
    print(f"Dataset size: {len(dataset)}")

    ## Put in to dataloader format
    dataloader = []

    # get cause_pairs (target_attr_prompt)
    cause_base_prompts, cause_source_prompts = get_prompt_pairs(
        dataset=dataset,
        base_attribute=attribute,
        source_attribute=attribute,
        n_interventions=eval_config.num_pairs_per_attribute,  # //2
    )

    # TODO:get isolation pairs (iso_prompt)
    # shuffle

    formatted_cause_pairs = []
    for base, source in zip(cause_base_prompts, cause_source_prompts):
        base_tokens_L = base.input_ids
        base_attn_mask_L = base.attention_mask
        base_pos = base.final_entity_token_pos
        base_pred = base.first_generated_token_id
        source_tokens_L = source.input_ids
        source_attn_mask_L = source.attention_mask
        source_pos = source.final_entity_token_pos
        source_pred = source.first_generated_token_id
        formatted_cause_pairs.append(
            (
                base_tokens_L,
                source_tokens_L,
                base_attn_mask_L,
                source_attn_mask_L,
                base_pos,
                source_pos,
                base_pred,
                source_pred,
            )
        )

    first_pair = formatted_cause_pairs[0]
    # print(f'first pair: {first_pair}')
    num_das_batches = len(formatted_cause_pairs) // eval_config.llm_batch_size
    for batch_idx in range(num_das_batches):
        batch_start = batch_idx * eval_config.llm_batch_size
        batch_end = batch_start + eval_config.llm_batch_size
        batch_data = formatted_cause_pairs[batch_start:batch_end]

        base_tokens_BL = []
        source_tokens_BL = []
        base_attn_mask_BL = []
        source_attn_mask_BL = []
        base_pos_B = []
        source_pos_B = []
        base_pred_B = []
        source_pred_B = []
        for (
            base_tokens_L,
            source_tokens_L,
            base_attn_mask_L,
            source_attn_mask_L,
            base_pos,
            source_pos,
            base_pred,
            source_pred,
        ) in batch_data:
            base_tokens_BL.append(base_tokens_L)
            source_tokens_BL.append(source_tokens_L)
            base_attn_mask_BL.append(base_attn_mask_L)
            source_attn_mask_BL.append(source_attn_mask_L)
            base_pos_B.append(base_pos)
            source_pos_B.append(source_pos)
            base_pred_B.append(base_pred)
            source_pred_B.append(source_pred)

        base_tokens_BL = torch.stack(base_tokens_BL).to(model.device)
        source_tokens_BL = torch.stack(source_tokens_BL).to(model.device)
        base_attn_mask_BL = torch.stack(base_attn_mask_BL).to(model.device)
        source_attn_mask_BL = torch.stack(source_attn_mask_BL).to(model.device)
        base_pos_B = torch.tensor(base_pos_B).to(model.device)
        source_pos_B = torch.tensor(source_pos_B).to(model.device)
        base_pred_B = torch.tensor(base_pred_B).to(model.device)
        source_pred_B = torch.tensor(source_pred_B).to(model.device)

        base_encoding_BL = BatchEncoding(
            {
                "input_ids": base_tokens_BL,
                "attention_mask": base_attn_mask_BL,
            }
        )
        source_encoding_BL = BatchEncoding(
            {
                "input_ids": source_tokens_BL,
                "attention_mask": source_attn_mask_BL,
            }
        )

        print(f"batch_encoding_input_id_shape: {base_encoding_BL['input_ids'].shape}")
        print(
            f"batch_encoding_attention_mask_shape: {base_encoding_BL['attention_mask'].shape}"
        )
        print(
            f"source_encoding_input_id_shape: {source_encoding_BL['input_ids'].shape}"
        )
        print(
            f"source_encoding_attention_mask_shape: {source_encoding_BL['attention_mask'].shape}"
        )
        dataloader.append(
            (
                base_encoding_BL,
                source_encoding_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
            )
        )

    print(f"loaded {len(dataloader)} batches")

    # inspect_dataloader(dataloader, tokenizer, n=1, view_strs=True)

    # Train MDBM
    mdbm.train_mdbm(
        model,
        tokenizer,
        eval_config,
        sae,
        train_loader=dataloader,
        val_loader=dataloader,
    )


if __name__ == "__main__":
    main()
