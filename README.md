# SAE Bench

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Running Evaluations](#running-evaluations)
- [Custom SAE Usage](#custom-sae-usage)
- [Training Your Own SAEs](#training-your-own-saes)
- [Graphing Results](#graphing-results)

CURRENT REPO STATUS: SAE Bench is currently a beta release. This repo is still under development as we clean up some of the rough edges left over from the research process. However, it is usable in the current state for both SAE Lens SAEs and custom SAEs.

## Overview

SAE Bench is a comprehensive suite of 8 evaluations for Sparse Autoencoder (SAE) models:

- **[Feature Absorption](https://arxiv.org/abs/2409.14507)**
- **[AutoInterp](https://blog.eleuther.ai/autointerp/)**
- **L0 / Loss Recovered**
- **[RAVEL](https://arxiv.org/abs/2402.17700) (under development)**
- **[Spurious Correlation Removal (SCR)](https://arxiv.org/abs/2411.18895)**
- **[Targeted Probe Pertubation (TPP)](https://arxiv.org/abs/2411.18895)**
- **Sparse Probing**
- **[Unlearning](https://arxiv.org/abs/2410.19278)**

For more information, refer to our [blog post](https://www.neuronpedia.org/sae-bench/info).

### Supported Models and SAEs

- **SAE Lens Pretrained SAEs**: Supports evaluations on any [SAE Lens](https://github.com/jbloomAus/SAELens) SAE.
- **dictionary_learning SAES**: We support evaluations on any SAE trained with the [dictionary_learning repo](https://github.com/saprmarks/dictionary_learning) (see [Custom SAE Usage](#custom-sae-usage)).
- **Custom SAEs**: Supports any general SAE object with `encode()` and `decode()` methods (see [Custom SAE Usage](#custom-sae-usage)).

### Installation

Set up a virtual environment with python >= 3.10.

```
git clone https://github.com/adamkarvonen/SAEBench.git
cd SAEBench
pip install -e .
```

Alternative, you can install from pypi:

```
pip install sae-bench
```

If you encounter dependency issues, you can use our tested working versions by uncommenting the fixed versions in pyproject.toml. All evals can be ran with current batch sizes on Gemma-2-2B on a 24GB VRAM GPU (e.g. a RTX 3090). By default, some evals cache LLM activations, which can require up to 100 GB of disk space. However, this can be disabled.

Autointerp requires the creation of `openai_api_key.txt`. Unlearning requires requesting access to the WMDP bio dataset (refer to `unlearning/README.md`).

## Getting Started

We recommend to get starting by going through the `sae_bench_demo.ipynb` notebook. In this notebook, we load both a custom SAE and an SAE Lens SAE, run both of them on multiple evaluations, and plot graphs of the results.

## Running Evaluations with SAE Lens

Each evaluation has an example command located in its respective `main.py` file. To run all evaluations on a selection of SAE Lens SAEs, refer to `shell_scripts/README.md`. Here's an example of how to run a sparse probing evaluation on a single SAE Bench Pythia-70M SAE:

```
python -m sae_bench.evals.sparse_probing.main \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped
```

The results will be saved to the `eval_results/sparse_probing` directory.

We use regex patterns to select SAE Lens SAEs. For more examples of regex patterns, refer to `sae_regex_selection.ipynb`.

Every eval folder contains an `eval_config.py`, which contains all relevant hyperparamters for that evaluation. The values are currently set to the default recommended values.

## Custom SAE Usage

Our goal is to have first class support for custom SAEs as the field is rapidly evolving. Our evaluations can run on any SAE object with `encode()`, `decode()`, and a few config values. We recommend referring to `sae_bench_demo.ipynb`. In this notebook, we load a custom SAE and an SAE Bench baseline SAE, run them on two evals, and graph the results. There is additional information about custom SAE usage in `sae_bench/custom_saes/README.md`.

If your SAEs are trained with the [dictionary_learning repo](https://github.com/saprmarks/dictionary_learning), you can evaluate your SAEs by passing in the name of the HuggingFace repo containing your SAEs. Refer to `sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py`.

For other SAE types, refer to `sae_bench/custom_saes/run_all_evals_custom_saes.py`.

We currently have a suite of SAE Bench SAEs on layer 8 of Pythia-160M and layer 12 of Gemma-2-2B, each trained on 500M tokens with some having checkpoints at various points. These SAEs can serve as baselines for any new custom SAEs. We also have baseline eval results, saved [here](https://huggingface.co/datasets/adamkarvonen/sae_bench_results_0125). For more information, refer to `sae_bench/custom_saes/README.md`.

## Training Your Own SAEs

You can deterministically replicate the training of our SAEs using scripts provided [here](https://github.com/adamkarvonen/dictionary_learning_demo), or implement your own SAE, or make a change to one of our SAE implementations. Once you train your new version, you can benchmark against our existing SAEs for a true apples to apples comparison.

## Graphing Results

If evaluating your own SAEs, we recommend using the graphing cells in `sae_bench_demo.ipynb`. To replicate all SAE Bench plots, refer to `graphing.ipynb`. In this notebook, we download all SAE Bench data and create a variety of plots.

## Computational Requirements

The computational requirements for running SAEBench evaluations were measured on an NVIDIA RTX 3090 GPU using 16K width SAEs trained on the Gemma-2-2B model. The table below breaks down the timing for each evaluation type into two components: an initial setup phase and the per-SAE evaluation time.

- **Setup Phase**: Includes operations like precomputing model activations, training probes, or other one-time preprocessing steps which can be reused across multiple SAE evaluations.
- **Per-SAE Evaluation Time**: The time required to evaluate a single SAE once the setup is complete.

The total evaluation time for a single SAE across all benchmarks is approximately **65 minutes**, with an additional **107 minutes** of setup time. Note that actual runtimes may vary significantly based on factors such as SAE dictionary size, base model, and GPU selection.

| Evaluation Type | Avg Time per SAE (min) | Setup Time (min) |
| --------------- | ---------------------- | ---------------- |
| Absorption      | 26                     | 33               |
| Core            | 9                      | 0                |
| SCR             | 6                      | 22               |
| TPP             | 2                      | 5                |
| Sparse Probing  | 3                      | 15               |
| Auto-Interp     | 9                      | 0                |
| Unlearning      | 10                     | 33               |
| **Total**       | **65**                 | **107**          |

## Development

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

To install the development dependencies, run:

```
poetry install
```

### Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. To run linting, run:

```
make lint
```

To run formatting, run:

```
make format
```

To run type checking, run:

```
make check-type
```

### Testing

Unit tests can be run with:

```
poetry run pytest tests/unit
```

These test will be run automatically on every PR in CI.

There are also acceptance tests than can be run with:

```
poetry run pytest tests/acceptance
```

These tests are expensive and will not be run automatically in CI, but are worth running manually before large changes.

### Running all CI checks locally

Before submitting a PR, run:

```
make check-ci
```

This will run linting, formatting, type checking, and unit tests. If these all pass, your PR should be good to go!

### Configuring VSCode for auto-formatting

If you use VSCode, install the Ruff plugin, and add the following to your `.vscode/settings.json` file:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

### Pre-commit hook

There's a pre-commit hook that will run ruff and pyright on each commit. To install it, run:

```bash
poetry run pre-commit install
```

### Updating Eval Output Schemas

Eval output structures / data types are under the `eval_output.py` file in each eval directory. If any of the `eval_output.py` files are updated, it's a good idea to run `python sae_bench/evals/generate_json_schemas.py` to make the json schemas match them as well.
