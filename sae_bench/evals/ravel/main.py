import argparse
import gc
import os
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime

import torch
from sae_lens import SAE
from tqdm import tqdm

import torch
from transformers import BatchEncoding, AutoTokenizer, AutoModelForCausalLM
import sae_lens
import random

import sae_bench.evals.ravel.mdbm as mdbm
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.instance import create_filtered_dataset
from sae_bench.evals.ravel.intervention import get_prompt_pairs
from sae_bench.evals.ravel.eval_output import (
    RAVELMetricCategories,
    RAVELMetricResults,
    RAVELEvalOutput,
    EVAL_TYPE_ID_RAVEL,
)


import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
)

LLM_NAME_MAP = {"gemma-2-2b": "google/gemma-2-2b"}


def create_dataloaders(
    cause_base_prompts,
    cause_source_prompts,
    iso_base_prompts,
    iso_source_prompts,
    model,
    eval_config,
    train_test_split,
):
    """
    Create train and validation dataloaders from prompt pairs.

    Args:
        cause_base_prompts: List of base prompts
        cause_source_prompts: List of source prompts
        model: The model (used for device information)
        eval_config: Configuration for evaluation
        train_test_split: Ratio of data to use for training (default: 0.5)

    Returns:
        train_loader: Dataloader for training
        val_loader: Dataloader for validation
    """
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

    formatted_iso_pairs = []
    for base, source in zip(iso_base_prompts, iso_source_prompts):
        base_tokens_L = base.input_ids
        base_attn_mask_L = base.attention_mask
        base_pos = base.final_entity_token_pos
        base_pred = base.first_generated_token_id
        source_tokens_L = source.input_ids
        source_attn_mask_L = source.attention_mask
        source_pos = source.final_entity_token_pos
        source_pred = source.first_generated_token_id
        formatted_iso_pairs.append(
            (
                base_tokens_L,
                source_tokens_L,
                base_attn_mask_L,
                source_attn_mask_L,
                base_pos,
                source_pos,
                base_pred,
                base_pred,  # This is the only difference for iso pairs - we don't want this to change
            )
        )

    all_formatted_pairs = formatted_cause_pairs + formatted_iso_pairs
    random.shuffle(all_formatted_pairs)

    # Split into train and validation sets
    total_pairs = len(all_formatted_pairs)
    train_size = int(total_pairs * train_test_split)

    train_pairs = all_formatted_pairs[:train_size]
    val_pairs = all_formatted_pairs[train_size:]

    print(
        f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs"
    )

    # Create dataloaders
    train_loader = create_dataloader_from_pairs(train_pairs, model, eval_config)
    val_loader = create_dataloader_from_pairs(val_pairs, model, eval_config)

    return train_loader, val_loader


def create_dataloader_from_pairs(formatted_pairs, model, eval_config):
    """
    Create a dataloader from formatted prompt pairs.

    Args:
        formatted_pairs: List of formatted prompt pairs
        model: The model (used for device information)
        eval_config: Configuration for evaluation

    Returns:
        dataloader: List of batched data
    """
    dataloader = []
    num_batches = len(formatted_pairs) // eval_config.llm_batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * eval_config.llm_batch_size
        batch_end = batch_start + eval_config.llm_batch_size
        batch_data = formatted_pairs[batch_start:batch_end]

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

    print(f"Created dataloader with {len(dataloader)} batches")
    return dataloader


def run_eval_single_cause_attribute(
    dataset,
    cause_attribute: str,
    iso_attributes: list[str],
    config: RAVELEvalConfig,
    sae: SAE,
    model: AutoModelForCausalLM,
) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

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

    mdbm_results = mdbm.train_mdbm(
        model,
        tokenizer,
        config,
        sae,
        train_loader=train_loader,
        val_loader=val_loader,
        verbose=True,
    )

    return mdbm_results


def run_eval_single_dataset(
    entity_class: str,
    config: RAVELEvalConfig,
    sae: SAE,
    model: AutoModelForCausalLM,
) -> tuple[dict[str, float], dict]:
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility."""

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

    attributes = config.entity_attribute_selection[entity_class]

    results_dict = {"cause_score": [], "isolation_score": [], "disentangle_score": []}
    per_class_results_dict = {}

    for cause_attribute in attributes:
        iso_attributes = [attr for attr in attributes if attr != cause_attribute]

        mdbm_results = run_eval_single_cause_attribute(
            dataset,
            cause_attribute,
            iso_attributes,
            config,
            sae,
            model,
        )

        results_dict["cause_score"].append(mdbm_results["cause_score"])
        results_dict["isolation_score"].append(mdbm_results["isolation_score"])
        results_dict["disentangle_score"].append(mdbm_results["disentangle_score"])

        per_class_results_dict[f"{entity_class}_{cause_attribute}"] = mdbm_results

    for key in results_dict.keys():
        results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

    return results_dict, per_class_results_dict


def run_eval_single_sae(
    config: RAVELEvalConfig,
    sae: SAE,
    model: AutoModelForCausalLM,
    device: str,
    artifacts_folder: str,
) -> tuple[dict[str, float | dict[str, float]], dict]:
    """hook_point: str is transformer lens format. example: f'blocks.{layer}.hook_resid_post'
    By default, we save activations for all datasets, and then reuse them for each sae.
    This is important to avoid recomputing activations for each SAE, and to ensure that the same activations are used for all SAEs.
    However, it can use 10s of GBs of disk space."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    os.makedirs(artifacts_folder, exist_ok=True)

    results_dict = {}

    dataset_results = {}
    per_class_dict = {}
    for entity_class in config.entity_attribute_selection.keys():
        (
            dataset_results[f"{entity_class}_results"],
            per_class_dict[f"{entity_class}_results"],
        ) = run_eval_single_dataset(
            entity_class,
            config,
            sae,
            model,
        )

    results_dict = general_utils.average_results_dictionaries(
        dataset_results, list(config.entity_attribute_selection.keys())
    )

    for entity_class, dataset_result in dataset_results.items():
        results_dict[f"{entity_class}"] = dataset_result

    return results_dict, per_class_dict  # type: ignore


def run_eval(
    config: RAVELEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    artifacts_path: str = "artifacts",
):
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)

    If clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    You may want to use this because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    artifacts_folder = None
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    config.model_name = LLM_NAME_MAP[config.model_name]

    if "gemma" in config.model_name:
        model_kwargs = {"attn_implementation": "eager"}
    else:
        model_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        **model_kwargs,
    )

    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        sae_id, sae, sparsity = general_utils.load_and_format_sae(
            sae_release, sae_object_or_id, device
        )  # type: ignore
        sae = sae.to(device=device, dtype=llm_dtype)

        sae_result_path = general_utils.get_results_filepath(
            output_path, sae_release, sae_id
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} as results already exist")
            continue

        artifacts_folder = os.path.join(
            artifacts_path,
            EVAL_TYPE_ID_RAVEL,
            config.model_name,
            sae.cfg.hook_name,
        )

        eval_results, per_class_dict = run_eval_single_sae(
            config,
            sae,
            model,
            device,
            artifacts_folder,
        )
        eval_output = RAVELEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=RAVELMetricCategories(
                ravel=RAVELMetricResults(
                    disentanglement_score=eval_results["disentangle_score"],
                    cause_score=eval_results["cause_score"],
                    isolation_score=eval_results["isolation_score"],
                )
            ),
            eval_result_details=[],
            eval_result_unstructured=per_class_dict,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=asdict(sae.cfg),
        )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


# def create_config_and_selected_saes(
#     args,
# ) -> tuple[SparseProbingEvalConfig, list[tuple[str, str]]]:
#     config = SparseProbingEvalConfig(
#         model_name=args.model_name,
#     )

#     if args.llm_batch_size is not None:
#         config.llm_batch_size = args.llm_batch_size
#     else:
#         config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
#             config.model_name
#         ]

#     if args.llm_dtype is not None:
#         config.llm_dtype = args.llm_dtype
#     else:
#         config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     if args.sae_batch_size is not None:
#         config.sae_batch_size = args.sae_batch_size

#     if args.random_seed is not None:
#         config.random_seed = args.random_seed

#     if args.lower_vram_usage:
#         config.lower_vram_usage = True

#     selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
#     assert len(selected_saes) > 0, "No SAEs selected"

#     releases = set([release for release, _ in selected_saes])

#     print(f"Selected SAEs from releases: {releases}")

#     for release, sae in selected_saes:
#         print(f"Sample SAEs: {release}, {sae}")

#     return config, selected_saes


# def arg_parser():
#     parser = argparse.ArgumentParser(description="Run sparse probing evaluation")
#     parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
#     parser.add_argument("--model_name", type=str, required=True, help="Model name")
#     parser.add_argument(
#         "--sae_regex_pattern",
#         type=str,
#         required=True,
#         help="Regex pattern for SAE selection",
#     )
#     parser.add_argument(
#         "--sae_block_pattern",
#         type=str,
#         required=True,
#         help="Regex pattern for SAE block selection",
#     )
#     parser.add_argument(
#         "--output_folder",
#         type=str,
#         default="eval_results/sparse_probing",
#         help="Output folder",
#     )
#     parser.add_argument(
#         "--force_rerun", action="store_true", help="Force rerun of experiments"
#     )
#     parser.add_argument(
#         "--llm_batch_size",
#         type=int,
#         default=None,
#         help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
#     )
#     parser.add_argument(
#         "--llm_dtype",
#         type=str,
#         default=None,
#         choices=[None, "float32", "float64", "float16", "bfloat16"],
#         help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
#     )
#     parser.add_argument(
#         "--sae_batch_size",
#         type=int,
#         default=None,
#         help="Batch size for SAE. If None, will be populated using default config value",
#     )
#     parser.add_argument(
#         "--lower_vram_usage",
#         action="store_true",
#         help="Lower GPU memory usage by doing more computation on the CPU. Useful on 1M width SAEs. Will be slower and require more system memory.",
#     )
#     parser.add_argument(
#         "--artifacts_path",
#         type=str,
#         default="artifacts",
#         help="Path to save artifacts",
#     )

#     return parser


# if __name__ == "__main__":
#     """
#     python -m sae_bench.evals.ravel.main \
#     --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
#     --sae_block_pattern "blocks.12.hook_resid_post__trainer_2" \
#     --model_name gemma-2-2b


#     """
#     args = arg_parser().parse_args()
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     config, selected_saes = create_config_and_selected_saes(args)

#     print(selected_saes)

#     # create output folder
#     os.makedirs(args.output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         args.output_folder,
#         args.force_rerun,
#         artifacts_path=args.artifacts_path,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import sae_bench.custom_saes.identity_sae as identity_sae
#     import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae

#     """
#     python evals/sparse_probing/main.py
#     """
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results/sparse_probing"

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = SparseProbingEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         output_folder,
#         force_rerun=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
