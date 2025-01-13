This repo implements David Chanin's feature absorption metric, with the absorption fraction metric added by Demian Till.

The code produces two scores:
- `mean_absorption_fraction_score` captures both full and partial absorption with an arbitrary number of absorbing latents. For a given SAE input, the absorption fraction is essentially the fraction of the SAE reconstruction's projection onto the ground truth probe activation that is not accounted for by the main latents which usually represent the feature in question.
- `mean_full_absorption_score` captures full absorption (not partial absorption) with a single absorbing latent. For a given SAE input, full absorption is judged to occur when the feature is present according to the ground truth probe, the main latents usually representing that feature have zero activation, and another latent compensates with a projection onto the ground truth probe direction which is above a set threshold as a proportion of the ground truth probe activation.

Estimated runtime:

- Pythia-70M: ~1 minute to collect activations / train probes per layer with SAEs, plus ~1 minute per SAE
- Gemma-2-2B: ~30 minutes to collect activations / train probes per layer with SAEs, plus ~10 minutes per SAE

Using Gemma-2-2B, at current batch sizes, I see a peak GPU memory usage of 24 GB. It successfully fits on an RTX 3090.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `evals/absorption/` and run `python main.py`. It should produce `eval_results/absorption/pythia-70m-deduped_layer_4_eval_results.json`.

`tests/test_absorption.py` contains an end-to-end test of the sparse probing eval. Expected results are in `tests/test_data/absorption_expected_results.json`. Running `pytest -s tests/test_absorption` will verify that the actual results are within the specified tolerance of the expected results.