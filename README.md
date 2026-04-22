# Coherence-Regularized Playlist Continuation

Decoder-only transformer for automatic playlist continuation (APC) on the
Spotify Million Playlist Dataset (MPD). The main research question is whether
adding a co-occurrence-based coherence regularization term to the training
objective improves playlist quality without unduly harming next-track accuracy
(nDCG@k).

## Dataset

Download the MPD from
[AICrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
Despite what the page says, the dataset can still be downloaded after free
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

The requirements are: `torch`, `numpy`, `pandas`, `tqdm`, `ipywidgets`, `scipy`.

## Training

`scripts/train.py` sweeps a list of coherence weights (lambda values), trains a
fresh model for each one, and saves results to a JSON file alongside the
checkpoints.

**Quick smoke-test** (runs in under a minute):

```bash
python scripts/train.py --max_train_playlists 128 --num_epochs 5
```

**Full sweep** (the settings used for the paper):

```bash
python scripts/train.py \
    --max_train_playlists 8000 \
    --num_epochs 20 \
    --d_model 256 --n_heads 4 --n_layers 4 --d_ff 512 \
    --coherence_weights 0.0 5.0 7.5 10.0 12.5 15.0 \
    --output_dir saved_models/sweep_8k
```

**Coherence mode ablation** (sequential vs. prefix_mean):

```bash
python scripts/train.py \
    --max_train_playlists 8000 \
    --num_epochs 20 \
    --d_model 256 --n_heads 4 --n_layers 4 --d_ff 512 \
    --coherence_mode prefix_mean \
    --coherence_weights 0.0 5.0 10.0 \
    --output_dir saved_models/prefix_mean_ablation
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--max_train_playlists` | 5000 | Number of training playlists |
| `--coherence_weights` | 0.0 0.01 ... 1.0 | Lambda values to sweep |
| `--coherence_mode` | `sequential` | `sequential`, `prefix_mean`, or `combined` |
| `--coherence_alpha` | 0.7 | Weight of sequential in `combined` mode |
| `--d_model / --n_layers / --d_ff` | 128 / 2 / 256 | Model size |
| `--num_epochs` | 20 | Epochs per lambda value |
| `--output_dir` | `saved_models/pareto_sweep` | Where to save checkpoints and results |
| `--log_dir` | `logs/` | Streaming per-run JSON logs (fault-tolerant) |

Each completed run is also streamed to `--log_dir` as a JSON file immediately
after it finishes, so partial results survive if the sweep crashes mid-way.

## Analysis

`scripts/analyze_results.py` reads a saved `results.json` and prints two
analyses: an nDCG@k breakdown (with per-rank deltas relative to the lambda=0
baseline) and a prediction diversity analysis (unique-track coverage, entropy,
Gini coefficient, and Spearman rank-frequency correlation with training
popularity).

```bash
# both analyses
python scripts/analyze_results.py \
    --results_dir saved_models/sweep_8k

# nDCG breakdown only (no checkpoint loading, runs instantly)
python scripts/analyze_results.py \
    --results_dir saved_models/sweep_8k \
    --skip_diversity

# diversity for a subset of lambda values
python scripts/analyze_results.py \
    --results_dir saved_models/sweep_8k \
    --lambdas 0.0 5.0 15.0
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
    train.py               # Pareto sweep training script
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
