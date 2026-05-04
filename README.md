# Coherence-Regularized Playlist Continuation

Decoder-only transformer for automatic playlist continuation (APC) on the
Spotify Million Playlist Dataset (MPD). The main research question is whether
adding a co-occurrence-based coherence regularization term to the training
objective improves playlist quality without unduly harming next-track accuracy
(nDCG@k).

## Dataset

Download the MPD from
[AICrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
The dataset can be downloaded after free
registration. Unzip it so that the repo root contains:

```
datasets/MPD/
    data/          # 1000 JSON slice files
    src/
    license.txt
    md5sums
    README.md
    stats.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The requirements are: `torch`, `numpy`, `pandas`, `tqdm`, `ipywidgets`, `scipy`. `matplotlib` enables plotting of results but is not required to run the experiment.

## Training

`scripts/train.py` sweeps a list of coherence weights (lambda values) over
multiple random seeds, trains a fresh model for each (lambda, seed) pair, and
saves both per-seed results and mean +/- std averaged across seeds to a JSON file
alongside the checkpoints.

**Quick test** (two seeds, five epochs):

```bash
python scripts/train.py --max_train_playlists 128 --num_epochs 5 --seeds 0 1
```

**Full sweep** (13-lambda grid, 10 seeds; the settings used for the paper):

```bash
python scripts/train.py \
    --max_train_playlists 8000 \
    --num_epochs 20 \
    --d_model 256 --n_heads 4 --n_layers 4 --d_ff 512 \
    --output_dir saved_models/final_sweep
```

The default lambda grid is
`0.0 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 15.0 20.0 25.0 50.0`
and the default seeds are `0` through `9`, so no extra flags are needed for the full run.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--max_train_playlists` | 8000 | Number of training playlists |
| `--coherence_weights` | 0.0 0.05 … 50.0 (13 values) | Lambda values to sweep |
| `--seeds` | 0 1 2 … 9 | Training seeds; one model trained per (lambda, seed) |
| `--seed` | 10 | Seed for data loading and train/val/test splits (fixed) |
| `--d_model / --n_layers / --d_ff` | 256 / 4 / 512 | Model size |
| `--num_epochs` | 20 | Epochs per (lambda, seed) |
| `--output_dir` | `saved_models/pareto_sweep` | Where to save checkpoints and results |
| `--log_dir` | `logs/` | Streaming per-run JSON logs (fault-tolerant) |

Each completed (lambda, seed) run is streamed to `--log_dir` immediately after
it finishes, so partial results survive a mid-sweep crash. `results.json`
contains per-seed results under `runs[i].seed_runs` and averaged metrics under
`runs[i].mean` / `runs[i].std`.

## Plotting

`scripts/plot_sweep.py` reads `results.json` and writes
figures to `<results_dir>/figures/` as both PDF and PNG.

All figures:

```bash
python scripts/plot_sweep.py --results_dir saved_models/final_sweep
```

Best checkpoint only:

```bash
python scripts/plot_sweep.py --results_dir saved_models/final_sweep --ckpt best
```

Pin specific lambda values for the training-curves panel:

```bash
python scripts/plot_sweep.py --results_dir saved_models/final_sweep \
    --select_lambdas 0 1 5 15 50
```

**Output files** (per checkpoint type `{best,final}`):

| File | Contents |
|---|---|
| `pareto_frontier_{best,final}.pdf` | Coherence vs nDCG@k scatter; λ-coloured points, error bars on both axes |
| `metric_sweep_{best,final}.pdf` | Three-panel: nDCG@k, coherence, and grad norm vs λ with ±1 σ bands |
| `training_curves.pdf` | Val nDCG and coherence over epochs for ~6 representative λ values |

All figures target a two-column paper format (double column = 7.16 in, 300 dpi).

## Analysis

`scripts/analyze_results.py` reads a saved `results.json` and prints two
analyses: an nDCG@k breakdown (with per-rank deltas relative to the lambda=0
baseline) and a prediction diversity analysis (unique-track coverage, entropy,
Gini coefficient, and Spearman rank-frequency correlation with training
popularity).

```bash
python scripts/analyze_results.py \
    --results_dir saved_models/final_sweep
```

Diversity metrics are saved to `results_dir/diversity_metrics.json`.

## Project structure

```
modules/
    coherence/
        cooccurence.py     # co-occurrence store and similarity matrix
        losses.py          # CE + coherence combined loss
    data_loading/mpd/
        reader.py          # streaming MPD JSON reader, MPDConfig
        vocab.py           # track vocabulary builder
        encoding.py        # playlist -> integer sequence encoder
        dataset.py         # PyTorch Dataset
        make_datasets.py   # assembles train/val/test DataLoaders
    models/
        decode_only_transformer.py   # GPT-style model
    utilities/
        logging.py         # structured JSON logging

scripts/
    train.py               # Pareto sweep training script (multi-seed)
    plot_sweep.py          # publication figures from results.json
    analyze_results.py     # post-training nDCG and diversity analysis

notebooks/
    midpoint_check.ipynb   # exploratory analysis and sanity checks

saved_models/              # checkpoints and results.json files (gitignored)
logs/                      # per-run streaming logs (gitignored)
reports/                   # paper drafts
```

## Saved results

The `saved_models/small_sweep_8k_big_model/` directory contains results from
the main experiment: a 10M-parameter model trained on 8K playlists with lambda
values 0.0, 5.0, 7.5, 10.0, 12.5, 15.0 (20 epochs each). Key findings:

- The CE-only baseline (lambda=0) collapses to 21 unique predicted tracks out
  of 32K in the vocabulary.
- lambda=5 is the threshold where coherence regularization meaningfully
  competes with CE; above it, eval coherence saturates around 0.68.
- Coherence costs ~10 pp at nDCG@1 but only ~3.7 pp at nDCG@20, suggesting
  the model preserves track ordering while shifting the top-1 prediction.
