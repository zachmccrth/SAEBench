import argparse
import gc
import os
import pickle
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime

import einops
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_bench.evals import memory_utils as memory_utils

import sae_bench.evals.scr_and_tpp.dataset_creation as dataset_creation
import sae_bench.evals.sparse_probing.probe_training as probe_training
import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.dataset_info as dataset_info
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.eval_output import (
    EVAL_TYPE_ID_SCR,
    EVAL_TYPE_ID_TPP,
    ScrEvalOutput,
    ScrMetricCategories,
    ScrMetrics,
    ScrResultDetail,
    TppEvalOutput,
    TppMetricCategories,
    TppMetrics,
    TppResultDetail,
)
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

COLUMN2_VALS_LOOKUP = {
    "LabHC/bias_in_bios_class_set1": ("male", "female"),
    "canrager/amazon_reviews_mcauley_1and5": (1.0, 5.0),
}


@torch.no_grad()
def get_effects_per_class_precomputed_acts(
    sae: SAE,
    probe: probe_training.Probe,
    class_idx: str,
    precomputed_acts: dict[str, torch.Tensor],
    perform_scr: bool,
    sae_batch_size: int,
) -> torch.Tensor:
    inputs_train_BLD, labels_train_B = probe_training.prepare_probe_data(
        precomputed_acts, class_idx, perform_scr
    )

    assert inputs_train_BLD.shape[0] == len(labels_train_B)

    """
    System here is basically:
    
    primary_device -> Device needed for high performance computation. (CUDA if available, CPU otherwise)
    
    secondary_device -> If VRAM is available, GPU, otherwise CPU.
    
    """

    primary_device = sae.device
    dtype = sae.dtype

    #TODO hardcoded for now,
    secondary_device = "cpu"

    # Preallocate data to the primary device for faster training, they should run out of scope and be removed automatically
    # If too large, may need to be moved by batches (inside for loop)
    inputs_train_BLD = inputs_train_BLD.to(primary_device, dtype)
    labels_train_B = labels_train_B.to(primary_device, dtype)

    running_sum_pos_F = torch.zeros(
        sae.W_dec.data.shape[0], dtype=torch.float32, device=primary_device
    )
    running_sum_neg_F = torch.zeros(
        sae.W_dec.data.shape[0], dtype=torch.float32, device=primary_device
    )
    count_pos = 0
    count_neg = 0

    for i in range(0, inputs_train_BLD.shape[0], sae_batch_size):
        # This should remain a view of the original inputs_train_BLD
        activation_batch_BLD = inputs_train_BLD[i : i + sae_batch_size]
        labels_batch_B = labels_train_B[i : i + sae_batch_size]

        # Sum over embedding dimension to ensure that at least some neurons are firing (removes empty tokens)
        nonzero_acts_BL = (einops.reduce(activation_batch_BLD, "B L D -> B L", "sum") != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum").to(
            torch.float32
        )

        # This intensive calculation should occur on the primary device
        f_BLF = sae.encode(activation_batch_BLD)
        average_sae_acts_BF = (
                einops.reduce(
                    f_BLF * nonzero_acts_BL[:, :, None],
                    "B L F -> B F",
                    "sum"
                ).to(torch.float32) / nonzero_acts_B[:, None]
        )
        # # Remove asap as the scope continues for a while
        # del f_BLF
        # torch.cuda.empty_cache()

        # Separate positive and negative samples
        pos_mask = labels_batch_B == dataset_info.POSITIVE_CLASS_LABEL
        neg_mask = labels_batch_B == dataset_info.NEGATIVE_CLASS_LABEL

        # Accumulate sums in fp32
        #TODO Check to see if necessary to move to cpu
        running_sum_pos_F += einops.reduce(
            average_sae_acts_BF[pos_mask], "B F -> F", "sum"
        )
        running_sum_neg_F += einops.reduce(
            average_sae_acts_BF[neg_mask], "B F -> F", "sum"
        )

        count_pos += pos_mask.sum().item()
        count_neg += neg_mask.sum().item()

    # Calculate means in fp32
    average_pos_sae_acts_F = (
        running_sum_pos_F / count_pos if count_pos > 0 else running_sum_pos_F
    )
    average_neg_sae_acts_F = (
        running_sum_neg_F / count_neg if count_neg > 0 else running_sum_neg_F
    )

    # The decoder matrix can be very large, so we move it to the same device as the activations
    average_acts_F = (average_pos_sae_acts_F - average_neg_sae_acts_F).to(dtype)

    #TODO examine size of decoder weight (should be fine?)
    probe_weight_D = probe.net.weight.to(dtype=dtype, device=primary_device)
    decoder_weight_DF = sae.W_dec.data.T.to(dtype=dtype, device=primary_device)

    dot_prod_F = (probe_weight_D @ decoder_weight_DF).squeeze()

    if not perform_scr:
        # Only consider activations from the positive class
        average_acts_F.clamp_(min=0.0)

    effects_F = (average_acts_F * dot_prod_F).to(dtype=torch.float32)

    if perform_scr:
        effects_F = effects_F.abs()

    return effects_F


def get_all_node_effects_for_one_sae(
    sae: SAE,
    probes: dict[str, probe_training.Probe],
    chosen_class_indices: list[str],
    perform_scr: bool,
    indirect_effect_acts: dict[str, torch.Tensor],
    sae_batch_size: int,
) -> dict[str, torch.Tensor]:
    node_effects = {}
    for ablated_class_idx in chosen_class_indices:
        node_effects[ablated_class_idx] = get_effects_per_class_precomputed_acts(
            sae,
            probes[ablated_class_idx],
            ablated_class_idx,
            indirect_effect_acts,
            perform_scr,
            sae_batch_size,
        )

    return node_effects


def select_top_n_features(
    effects: torch.Tensor, n: int, class_name: str
) -> torch.Tensor:
    assert n <= effects.numel(), (
        f"n ({n}) must not be larger than the number of features ({effects.numel()}) for ablation class {class_name}"
    )

    # Find non-zero effects
    non_zero_mask = effects != 0
    non_zero_effects = effects[non_zero_mask]
    num_non_zero = non_zero_effects.numel()

    if num_non_zero < n:
        print(
            f"WARNING: only {num_non_zero} non-zero effects found for ablation class {class_name}, which is less than the requested {n}."
        )

    # Select top n or all non-zero effects, whichever is smaller
    k = min(n, num_non_zero)

    if k == 0:
        print(
            f"WARNING: No non-zero effects found for ablation class {class_name}. Returning an empty mask."
        )
        top_n_features = torch.zeros_like(effects, dtype=torch.bool)
    else:
        # Get the indices of the top N effects
        _, top_indices = torch.topk(effects, k)

        # Create a boolean mask tensor
        mask = torch.zeros_like(effects, dtype=torch.bool)
        mask[top_indices] = True

        top_n_features = mask

    return top_n_features


def ablated_precomputed_activations(
    ablation_acts_BLD: torch.Tensor,
    sae: SAE,
    to_ablate: torch.Tensor,
    sae_batch_size: int,
) -> torch.Tensor:
    """NOTE: We don't pass in the attention mask. Thus, we must have already zeroed out all masked tokens in ablation_acts_BLD."""

    all_acts_list_BD = []

    for i in range(0, ablation_acts_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = ablation_acts_BLD[i : i + sae_batch_size]
        dtype = activation_batch_BLD.dtype

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        f_BLF = sae.encode(activation_batch_BLD)
        x_hat_BLD = sae.decode(f_BLF)

        error_BLD = activation_batch_BLD - x_hat_BLD

        f_BLF[..., to_ablate] = 0.0  # zero ablation

        modified_acts_BLD = sae.decode(f_BLF) + error_BLD

        # Get the average activation per input. We divide by the number of nonzero activations for the attention mask
        probe_acts_BD = (
            einops.reduce(modified_acts_BLD, "B L D -> B D", "sum")
            / nonzero_acts_B[:, None]
        )
        all_acts_list_BD.append(probe_acts_BD)

    all_acts_BD = torch.cat(all_acts_list_BD, dim=0)

    return all_acts_BD


def get_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
    perform_scr: bool,
) -> dict[str, float]:
    test_accuracies = {}
    for class_name in all_class_list:
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, perform_scr=perform_scr
        )
        probe: probe_training.Probe = probes[class_name]
        test_acts = test_acts.to(device=probe.net.weight.device)
        test_labels = test_labels.to(device=probe.net.weight.device)
        test_acc_probe = probe_training.test_probe_gpu(
            test_acts,
            test_labels,
            probe_batch_size,
            probes[class_name],
        )
        test_accuracies[class_name] = test_acc_probe

    if perform_scr:
        scr_probe_accuracies = get_scr_probe_test_accuracy(
            probes, all_class_list, all_activations, probe_batch_size
        )
        test_accuracies.update(scr_probe_accuracies)

    return test_accuracies


def get_scr_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
) -> dict[str, float]:
    """Tests e.g. male_professor / female_nurse probe on professor / nurse labels"""
    test_accuracies = {}
    for class_name in all_class_list:
        if class_name not in dataset_info.PAIRED_CLASS_KEYS:
            continue
        spurious_class_names = [
            key for key in dataset_info.PAIRED_CLASS_KEYS if key != class_name
        ]
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, perform_scr=True
        )
        probe: probe_training.Probe = probes[class_name]
        test_acts = test_acts.to(device=probe.net.weight.device)
        test_labels = test_labels.to(device=probe.net.weight.device)
        for spurious_class_name in spurious_class_names:
            test_acc_probe = probe_training.test_probe_gpu(
                test_acts,
                test_labels,
                probe_batch_size,
                probes[spurious_class_name],
            )
            combined_class_name = f"{spurious_class_name} probe on {class_name} data"
            test_accuracies[combined_class_name] = test_acc_probe

    return test_accuracies


def perform_feature_ablations(
    probes: dict[str, probe_training.Probe],
    sae: SAE,
    sae_batch_size: int,
    all_test_acts_BLD: dict[str, torch.Tensor],
    node_effects: dict[str, torch.Tensor],
    top_n_values: list[int],
    chosen_classes: list[str],
    probe_batch_size: int,
    perform_scr: bool,
) -> dict[str, dict[int, dict[str, float]]]:
    ablated_class_accuracies = {}
    for ablated_class_name in chosen_classes:
        ablated_class_accuracies[ablated_class_name] = {}
        for top_n in top_n_values:
            selected_features_F = select_top_n_features(
                node_effects[ablated_class_name], top_n, ablated_class_name
            )
            test_acts_ablated = {}
            for evaluated_class_name in all_test_acts_BLD.keys():
                test_acts_ablated[evaluated_class_name] = (
                    ablated_precomputed_activations(
                        all_test_acts_BLD[evaluated_class_name],
                        sae,
                        selected_features_F,
                        sae_batch_size,
                    )
                )

            ablated_class_accuracies[ablated_class_name][top_n] = (
                get_probe_test_accuracy(
                    probes,
                    chosen_classes,
                    test_acts_ablated,
                    probe_batch_size,
                    perform_scr,
                )
            )
    return ablated_class_accuracies


def get_scr_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> dict[str, float]:
    """raw_results: dict[class_name][threshold][class_name] = float
    llm_clean_accs: dict[class_name] = float
    Returns: dict[metric_name] = float"""

    results = {}
    eval_probe_class_id = "male_professor / female_nurse"

    dirs = [1, 2]

    dir1_class_name = f"{eval_probe_class_id} probe on professor / nurse data"
    dir2_class_name = f"{eval_probe_class_id} probe on male / female data"

    dir1_acc = llm_clean_accs[dir1_class_name]
    dir2_acc = llm_clean_accs[dir2_class_name]

    for dir in dirs:
        if dir == 1:
            ablated_probe_class_id = "male / female"
            eval_data_class_id = "professor / nurse"
        elif dir == 2:
            ablated_probe_class_id = "professor / nurse"
            eval_data_class_id = "male / female"
        else:
            raise ValueError("Invalid dir.")

        for threshold in class_accuracies[ablated_probe_class_id]:
            clean_acc = llm_clean_accs[eval_data_class_id]

            combined_class_name = (
                f"{eval_probe_class_id} probe on {eval_data_class_id} data"
            )

            original_acc = llm_clean_accs[combined_class_name]

            changed_acc = class_accuracies[ablated_probe_class_id][threshold][
                combined_class_name
            ]

            if (clean_acc - original_acc) < 0.001:
                scr_score = 0
            else:
                scr_score = (changed_acc - original_acc) / (clean_acc - original_acc)

            print(
                f"dir: {dir}, original_acc: {original_acc}, clean_acc: {clean_acc}, changed_acc: {changed_acc}, scr_score: {scr_score}"
            )

            metric_key = f"scr_dir{dir}_threshold_{threshold}"

            results[metric_key] = scr_score

            scr_metric_key = f"scr_metric_threshold_{threshold}"
            if dir1_acc < dir2_acc and dir == 1:
                results[scr_metric_key] = scr_score
            elif dir1_acc > dir2_acc and dir == 2:
                results[scr_metric_key] = scr_score

    return results


def create_tpp_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]]]:
    """Calculates TPP metrics for each class and overall averages.

    Args:
        class_accuracies: Nested dict mapping class_name -> threshold -> other_class -> accuracy
        llm_clean_accs: Dict mapping class_name -> clean accuracy

    Returns:
        Tuple containing:
        - Dict mapping metric_name -> value for overall averages
        - Dict mapping class_name -> metric_name -> value
    """
    per_class_results = {}
    overall_results = {}
    classes = list(llm_clean_accs.keys())

    for class_name in classes:
        if " probe on " in class_name:
            raise ValueError("This is SCR, shouldn't be here.")

        class_metrics = {}
        intended_clean_acc = llm_clean_accs[class_name]

        # Calculate metrics for each threshold
        for threshold in class_accuracies[class_name]:
            # Intended differences
            intended_patched_acc = class_accuracies[class_name][threshold][class_name]
            intended_diff = intended_clean_acc - intended_patched_acc

            # Unintended differences for this threshold
            unintended_diffs = []
            for unintended_class in classes:
                if unintended_class == class_name:
                    continue

                unintended_clean_acc = llm_clean_accs[unintended_class]
                unintended_patched_acc = class_accuracies[class_name][threshold][
                    unintended_class
                ]
                unintended_diff = unintended_clean_acc - unintended_patched_acc
                unintended_diffs.append(unintended_diff)

            avg_unintended = sum(unintended_diffs) / len(unintended_diffs)
            avg_diff = intended_diff - avg_unintended

            # Store with original key format
            class_metrics[f"tpp_threshold_{threshold}_total_metric"] = avg_diff
            class_metrics[f"tpp_threshold_{threshold}_intended_diff_only"] = (
                intended_diff
            )
            class_metrics[f"tpp_threshold_{threshold}_unintended_diff_only"] = (
                avg_unintended
            )

        per_class_results[class_name] = class_metrics

    # Calculate overall averages across classes
    # First, determine all metric keys from the first class
    metric_keys = next(iter(per_class_results.values())).keys()

    for metric_key in metric_keys:
        values = [
            class_metrics[metric_key] for class_metrics in per_class_results.values()
        ]
        overall_results[metric_key] = sum(values) / len(values)

    return overall_results, per_class_results


def get_dataset_activations(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    model: HookedTransformer,
    llm_batch_size: int,
    layer: int,
    hook_point: str,
    device: str,
    chosen_classes: list[str],
    column1_vals: tuple[str, str] | None = None,
    column2_vals: tuple[str, str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_data, test_data = dataset_creation.get_train_test_data(
        dataset_name,
        config.perform_scr,
        config.train_set_size,
        config.test_set_size,
        config.random_seed,
        column1_vals,
        column2_vals,
    )

    if not config.perform_scr:
        train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
        test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data_dictionary(
        train_data,
        model.tokenizer,  # type: ignore
        config.context_length,
        device,
    )
    test_data = dataset_utils.tokenize_data_dictionary(
        test_data,
        model.tokenizer,  # type: ignore
        config.context_length,
        device,
    )

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data,
        model,
        llm_batch_size,
        layer,
        hook_point,
        mask_bos_pad_eos_tokens=True,
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data,
        model,
        llm_batch_size,
        layer,
        hook_point,
        mask_bos_pad_eos_tokens=True,
    )

    return all_train_acts_BLD, all_test_acts_BLD


def run_eval_single_dataset(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
    column1_vals: tuple[str, str] | None = None,
) -> tuple[dict[str, dict[str, dict[int, dict[str, float]]]], dict[str, float]]:
    """Return dict is of the form:
    dict[ablated_class_name][threshold][measured_acc_class_name] = float

    config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility."""

    column2_vals = COLUMN2_VALS_LOOKUP[dataset_name]

    if not config.perform_scr:
        chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]
        activations_filename = f"{dataset_name}_activations.pt".replace("/", "_")
        probes_filename = f"{dataset_name}_probes.pkl".replace("/", "_")
    else:
        chosen_classes = list(dataset_info.PAIRED_CLASS_KEYS.keys())
        activations_filename = f"{dataset_name}_{column1_vals[0]}_{column1_vals[1]}_activations.pt".replace(  # type: ignore
            "/", "_"
        )
        probes_filename = (
            f"{dataset_name}_{column1_vals[0]}_{column1_vals[1]}_probes.pkl".replace(  # type: ignore
                "/", "_"
            )
        )

    activations_path = os.path.join(artifacts_folder, activations_filename)
    probes_path = os.path.join(artifacts_folder, probes_filename)

    if not os.path.exists(activations_path):
        if config.lower_vram_usage:
            model = model.to(device)  # type: ignore
        all_train_acts_BLD, all_test_acts_BLD = get_dataset_activations(
            dataset_name,
            config,
            model,
            config.llm_batch_size,
            layer,
            hook_point,
            device,
            chosen_classes,
            column1_vals,
            column2_vals,
        )
        if config.lower_vram_usage:
            model = model.to("cpu")  # type: ignore

            memory_utils.move_dict_of_tensors_to_device(all_train_acts_BLD, "cpu")
            # memory_utils.move_dict_of_tensors_to_device(all_test_acts_BLD, "cpu")

            train_acts_size_bytes = 0
            for key in all_train_acts_BLD:
                train_acts_size_bytes += all_train_acts_BLD[key].nbytes

            train_acts_size_mb = train_acts_size_bytes / 1024 ** 2
            train_acts_size_gb = train_acts_size_bytes / 1024 ** 3

            # test_acts_size_bytes = 0
            # for key in all_test_acts_BLD:
            #     test_acts_size_bytes += all_test_acts_BLD[key].nbytes

            # test_acts_size_mb = test_acts_size_bytes / 1024 ** 2
            # test_acts_size_gb = test_acts_size_bytes / 1024 ** 3

            print(f"Storing training activations on cpu. Saved {train_acts_size_gb} GB from VRAM")
            # print(f"Storing test activations on cpu. Saved {test_acts_size_gb} GB from VRAM")




        all_meaned_train_acts_BD = (
            activation_collection.create_meaned_model_activations(all_train_acts_BLD)
        )
        all_meaned_test_acts_BD = activation_collection.create_meaned_model_activations(
            all_test_acts_BLD
        )

        torch.set_grad_enabled(True)

        #NOTE: This seems fast enough for now on the cpu (actually maybe it runs on the gpu?)

        llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
            all_meaned_train_acts_BD,
            all_meaned_test_acts_BD,
            select_top_k=None,
            use_sklearn=False,
            batch_size=config.probe_train_batch_size,
            epochs=config.probe_epochs,
            lr=config.probe_lr,
            perform_scr=config.perform_scr,
            early_stopping_patience=config.early_stopping_patience,
            l1_penalty=config.probe_l1_penalty,
        )

        torch.set_grad_enabled(False)


        #NOTE: This seems fast enough for now on the cpu
        llm_test_accuracies = get_probe_test_accuracy(
            llm_probes,  # type: ignore
            chosen_classes,
            all_meaned_test_acts_BD,
            config.probe_test_batch_size,
            config.perform_scr,
        )

        acts = {
            "train": all_train_acts_BLD,
            "test": all_test_acts_BLD,
        }

        llm_probes_dict = {
            "llm_probes": llm_probes,
            "llm_test_accuracies": llm_test_accuracies,
        }

        if save_activations:
            torch.save(acts, activations_path)
            with open(probes_path, "wb") as f:
                pickle.dump(llm_probes_dict, f)
    else:
        if config.lower_vram_usage:
            model = model.to("cpu")  # type: ignore
        print(f"Loading activations from {activations_path}")
        acts = torch.load(activations_path)
        all_train_acts_BLD= acts["train"]
        all_test_acts_BLD = acts["test"]

        print(f"Loading probes from {probes_path}")
        with open(probes_path, "rb") as f:
            llm_probes_dict = pickle.load(f)

        llm_probes = llm_probes_dict["llm_probes"]
        llm_test_accuracies = llm_probes_dict["llm_test_accuracies"]

    torch.set_grad_enabled(False)

    # Consider when memory should be handled?
    sae_node_effects = get_all_node_effects_for_one_sae(
        sae,
        llm_probes,  # type: ignore
        chosen_classes,
        config.perform_scr,
        all_train_acts_BLD,
        config.sae_batch_size,
    )

    ablated_class_accuracies = perform_feature_ablations(
        llm_probes,  # type: ignore
        sae,
        config.sae_batch_size,
        all_test_acts_BLD,
        sae_node_effects,
        config.n_values,
        chosen_classes,
        config.probe_test_batch_size,
        config.perform_scr,
    )

    return ablated_class_accuracies, llm_test_accuracies  # type: ignore


def run_eval_single_sae(
    config: ScrAndTppEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
) -> tuple[dict[str, float | dict[str, float]], dict]:
    """hook_point: str is transformer lens format. example: f'blocks.{layer}.hook_resid_post'
    By default, we save activations for all datasets, and then reuse them for each sae.
    This is important to avoid recomputing activations for each SAE, and to ensure that the same activations are used for all SAEs.
    However, it can use 10s of GBs of disk space."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    os.makedirs(artifacts_folder, exist_ok=True)

    dataset_results = {}
    per_dataset_class_results = {}

    averaging_names = []

    for dataset_name in config.dataset_names:
        if config.perform_scr:
            column1_vals_list = config.column1_vals_lookup[dataset_name]
            for column1_vals in column1_vals_list:
                run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
                raw_results, llm_clean_accs = run_eval_single_dataset(
                    dataset_name,
                    config,
                    sae,
                    model,
                    sae.cfg.hook_layer,
                    sae.cfg.hook_name,
                    device,
                    artifacts_folder,
                    save_activations,
                    column1_vals,
                )

                processed_results = get_scr_plotting_dict(raw_results, llm_clean_accs)  # type: ignore

                dataset_results[f"{run_name}_results"] = processed_results

                averaging_names.append(run_name)

        else:
            run_name = f"{dataset_name}_tpp"
            raw_results, llm_clean_accs = run_eval_single_dataset(
                dataset_name,
                config,
                sae,
                model,
                sae.cfg.hook_layer,
                sae.cfg.hook_name,
                device,
                artifacts_folder,
                save_activations,
            )

            processed_results, per_class_results = create_tpp_plotting_dict(
                raw_results,  # type: ignore
                llm_clean_accs,
            )
            dataset_results[f"{run_name}_results"] = processed_results
            per_dataset_class_results[dataset_name] = per_class_results

            averaging_names.append(run_name)

    results_dict = general_utils.average_results_dictionaries(
        dataset_results, averaging_names
    )
    results_dict.update(dataset_results)

    if config.lower_vram_usage:
        model = model.to(device)  # type: ignore

    return results_dict, per_dataset_class_results  # type: ignore


def run_eval(
    config: ScrAndTppEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
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

    if config.perform_scr:
        eval_type = EVAL_TYPE_ID_SCR
    else:
        eval_type = EVAL_TYPE_ID_TPP
    output_path = os.path.join(output_path, eval_type)
    os.makedirs(output_path, exist_ok=True)

    artifacts_folder = None

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
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
            eval_type,
            config.model_name,
            sae.cfg.hook_name,
        )

        scr_or_tpp_results, per_dataset_class_results = run_eval_single_sae(
            config,
            sae,
            model,
            device,
            artifacts_folder,
            save_activations,
        )
        if eval_type == EVAL_TYPE_ID_SCR:
            eval_output = ScrEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=ScrMetricCategories(
                    scr_metrics=ScrMetrics(
                        **{
                            k: v
                            for k, v in scr_or_tpp_results.items()
                            if not isinstance(v, dict)
                        }
                    )
                ),
                eval_result_details=[
                    ScrResultDetail(
                        dataset_name=dataset_name,
                        **result,
                    )
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
                sae_cfg_dict=asdict(sae.cfg),
            )
        elif eval_type == EVAL_TYPE_ID_TPP:
            eval_output = TppEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=TppMetricCategories(
                    tpp_metrics=TppMetrics(
                        **{
                            k: v
                            for k, v in scr_or_tpp_results.items()
                            if not isinstance(v, dict)
                        }
                    )
                ),
                eval_result_details=[
                    TppResultDetail(
                        dataset_name=dataset_name,
                        **result,
                    )
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                eval_result_unstructured=per_dataset_class_results,
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
                sae_cfg_dict=asdict(sae.cfg),
            )
        else:
            raise ValueError(f"Invalid eval type: {eval_type}")

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    if clean_up_activations:
        if artifacts_folder is not None and os.path.exists(artifacts_folder):
            shutil.rmtree(artifacts_folder)

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[ScrAndTppEvalConfig, list[tuple[str, str]]]:
    config = ScrAndTppEvalConfig(
        model_name=args.model_name,
        perform_scr=args.perform_scr,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    if args.lower_vram_usage:
        config.lower_vram_usage = True

    if args.sae_batch_size is not None:
        config.sae_batch_size = args.sae_batch_size

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR or TPP evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results",
        help="Output folder",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="artifacts",
        help="Path to save artifacts",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--clean_up_activations",
        action="store_true",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Save the generated LLM activations for later use",
    )

    def str_to_bool(value):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--perform_scr",
        type=str_to_bool,
        required=True,
        help="If true, do Spurious Correlation Removal (SCR). If false, do TPP.",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )
    parser.add_argument(
        "--sae_batch_size",
        type=int,
        default=None,
        help="Batch size for SAE. If None, will be populated using default config value",
    )
    parser.add_argument(
        "--lower_vram_usage",
        action="store_true",
        help="Lower GPU memory usage by moving model to CPU when not required. Useful on 1M width SAEs. Will be slower and require more system memory.",
    )

    return parser


if __name__ == "__main__":
    """
    Example pythia-70m usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped \
    --perform_scr true

    Example Gemma-2-2B SAE Bench usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name gemma-2-2b \
    --perform_scr true

    Example Gemma-2-2B Gemma-Scope usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_20/width_16k/average_l0_139" \
    --model_name gemma-2-2b \
    --perform_scr true
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
        args.clean_up_activations,
        args.save_activations,
        artifacts_path=args.artifacts_path,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import sae_bench.custom_saes.identity_sae as identity_sae
#     import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae

#     """
#     python evals/scr_and_tpp/main.py
#     """
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results"
#     perform_scr = True

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = ScrAndTppEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#         perform_scr=perform_scr,
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
#         clean_up_activations=False,
#         save_activations=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
