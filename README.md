<div align="center">

# L-FAME Benchmark

### **L**ongitudinal **F**ocus **A**ttention **M**editation **E**EG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-L--FAME-yellow)](https://huggingface.co/datasets/L-FAME-Dataset-Benchmark/L-FAME)
[![BIDS](https://img.shields.io/badge/Format-BIDS%201.7-green)](https://bids.neuroimaging.io/)

</div>

---

L-FAME is the **first longitudinal EEG benchmark** for evaluating neural decoding across meditation sessions, intervention types, and cultural groups. This quick-start package ships with a 9-subject sample dataset and all source code needed to reproduce the three benchmark tasks **end-to-end without any external data downloads**.

---

## Table of Contents

1. [Benchmark Overview](#benchmark-overview)
2. [Repository Structure](#repository-structure)
3. [Data Format](#data-format)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Running the Benchmark](#running-the-benchmark)
   - [Task 1 — Cognitive State Decoding](#task-1--cognitive-state-decoding)
   - [Task 2 — Meditation Technique Classification](#task-2--meditation-technique-classification)
   - [Task 3 — Cross-Session Adaptation](#task-3--cross-session-adaptation)
7. [CLI Reference](#cli-reference)
8. [Models](#models)
9. [Configuration](#configuration)
10. [Benchmark Results](#benchmark-results)
11. [References](#references)
12. [Citation](#citation)

---

## Benchmark Overview

| Task | Description | Strategies | N |
|------|-------------|-----------|---|
| **Task 1** | Cognitive State Decoding (Rest vs. Meditation) | Intra-subject · Inter-subject · LOSO | 74 |
| **Task 2 (Pre)** | Meditation Technique Classification (HK / SA / BF) | Inter-subject | 74 |
| **Task 2 (Post)** | Meditation Technique Classification (HK / SA / BF) | Inter-subject | 44 |
| **Task 3** | Cross-Session Adaptation (Pre-intervention → Post-intervention) | Zero-shot · Few-shot | 44 |

> *HK = Hare Krishna · SA = Satanama · BF = Breath-Focus. N varies due to longitudinal attrition (74 pre-intervention, 44 completed post-intervention).*

### Full Dataset

The complete 74-subject BIDS-formatted dataset is publicly available on Hugging Face:

> 🤗 **[https://huggingface.co/datasets/L-FAME-Dataset-Benchmark/L-FAME](https://huggingface.co/datasets/L-FAME-Dataset-Benchmark/L-FAME)**

The full dataset includes raw EEG in BrainVision format (`.eeg` / `.vhdr` / `.vmrk`), BIDS-compliant metadata (channels, events, electrode positions, coordinate systems), and pre-processed NumPy arrays under `derivatives/ml_preproc_data/`.

---

## Repository Structure

```
L-FAME_quickstart/
├── README.md
├── requirements.txt
│
├── config/
│   ├── task1_config.yaml        # Hyperparameters for Task 1
│   ├── task2_config.yaml        # Hyperparameters for Task 2
│   └── task3_config.yaml        # Hyperparameters for Task 3
│
├── src/
│   ├── dataset_api.py                  # Dataset class & dataloader builder
│   ├── main_benchmark_task1_task2.py   # Train / evaluate Tasks 1 & 2
│   ├── main_benchmark_task3.py         # Fine-tune / evaluate Task 3
│   └── models/
│       ├── eegnet.py            # EEGNet [1]
│       ├── shallowcnn.py        # ShallowConvNet [2]
│       ├── deepcnn.py           # DeepConvNet [2]
│       ├── eegconformer.py      # EEG Conformer [3]
│       ├── dgcnn.py             # DGCNN [4]
│       └── utils/               # Shared layers and transformer blocks
│
└── data/
    ├── participants.tsv                  # Subject metadata (9-subject sample)
    └── derivatives/
        └── ml_preproc_data/
            ├── sub-001/  ─── HK group   # 4 × .npy files per subject
            ├── sub-002/
            ├── sub-008/
            ├── sub-003/  ─── SA group
            ├── sub-004/
            ├── sub-006/
            ├── sub-005/  ─── BF group
            ├── sub-007/
            └── sub-009/
```

---

## Data Format

### Pre-processed Arrays (`.npy`)

Each `.npy` file contains a single continuous EEG recording stored as a **NumPy array**:

```
Shape : (n_channels, n_samples)  →  (64, T)
Dtype : float32
Units : µV (channel-wise z-scored per window at load time)
```

**File naming convention** (BIDS-derived):

```
sub-{ID}_ses-{SESSION}_task-{TASK}_eeg_preproc.npy

  {ID}      : zero-padded subject number, e.g. 001
  {SESSION} : ses-premedita  (pre-intervention)
              ses-posmedita  (post-intervention)
  {TASK}    : task-restCE01  (label 0 — eyes-closed resting state)
              task-slMedita  (label 1 — silent meditation)
```

**Files used per benchmark task:**

| Task | Session | Tasks loaded | Label |
|------|---------|-------------|-------|
| Task 1 | `ses-premedita` | `task-restCE01` | 0 — Rest |
| Task 1 | `ses-premedita` | `task-slMedita` | 1 — Meditation |
| Task 2 | `ses-premedita` or `ses-posmedita` | `task-slMedita` | 0/1/2 — HK/SA/BF |
| Task 3 | `ses-posmedita` | `task-restCE01` | 0 — Post-Rest |
| Task 3 | `ses-posmedita` | `task-slMedita` | 1 — Post-Meditation |

### Windowing

Windows are extracted on the fly by `dataset_api.py`. The overlap is strategy-dependent:

| Parameter | Task 1 intra · Task 3 | Task 1 inter / LOSO · Task 2 |
|-----------|:---:|:---:|
| Window length | 4.0 s (1 000 samples) | 4.0 s (1 000 samples) |
| Overlap | 3.5 s (87.5 %) | 2.0 s (50 %) |
| Stride | 0.5 s | 2.0 s |

> Dense overlap (3.5 s / 87.5 %) is used for intra-subject and cross-session evaluations to
> maximise the number of windows available per subject. Sparser overlap (2.0 s / 50 %) is
> sufficient for inter-subject and LOSO splits where many subjects contribute to the pool.

### Sample Dataset (this repository)

| Property | Value |
|----------|-------|
| Subjects | 9 — 3 per group (HK, SA, BF), all with both sessions |
| Duration per file | 120 s (truncated from full ~240 s; sufficient for all strategies) |
| Array shape | `(64, 30 000)` |
| Sampling rate | 250 Hz |
| Total size | ~264 MB (36 files × ~7.3 MB) |

**Subject groups included:**

| Group | Meditation style | Subjects (sample) |
|-------|-----------------|-------------------|
| HK | Hare Krishna mantra | sub-01, sub-02, sub-08 |
| SA | Satanama Kundalini | sub-03, sub-04, sub-06 |
| BF | Breath-Focus | sub-05, sub-07, sub-09 |

---

## Installation

```bash
# 1. Clone or download this repository
git clone https://github.com/<your-repo>/L-FAME_quickstart.git
cd L-FAME_quickstart

# 2. Create a dedicated conda environment (recommended)
conda create -n l-fame python=3.10 -y
conda activate l-fame

# 3. Install all dependencies
pip install -r requirements.txt
```

**Dependencies:** `torch>=2.0` · `numpy` · `pandas` · `scikit-learn` · `einops` · `mne` · `pyyaml` · `scipy`

> **GPU:** CUDA is used automatically when available. All scripts also run on CPU (slower but fully functional for the quick-start demo).

---

## Quick Start

Reproduce a complete Task 1 → Task 3 pipeline with three commands:

```bash
# Task 1 — intra-subject decoding for sub-01
python src/main_benchmark_task1_task2.py \
    --task 1 --model EEGNet --strategy intra_block \
    --subject sub-01 --config config/task1_config.yaml

# Task 2 — technique classification (pre-session)
python src/main_benchmark_task1_task2.py \
    --task 2 --model EEGNet --strategy inter \
    --session pre --config config/task2_config.yaml

# Task 3 — zero-shot cross-session adaptation for sub-01
#   (requires the Task 1 checkpoint created above)
python src/main_benchmark_task3.py \
    --model EEGNet --strategy zero_shot --subject sub-01 \
    --config config/task3_config.yaml \
    --task1_dir results/checkpoints/task1 --task1_intra intra_block
```

> All commands are run from the **`L-FAME_quickstart/`** directory.

---

## Running the Benchmark

### Task 1 — Cognitive State Decoding

Binary classification of resting-state (label 0) vs. silent meditation (label 1) using pre-intervention EEG.

#### Intra-subject — block split *(recommended)*

`intra_block` alternates train/test blocks throughout the recording (4:1 ratio), which
better reflects realistic within-session variability and consistently yields higher accuracy
than a single chronological cut. Use `intra_chrono` only when strict temporal ordering is
required by your experimental protocol.

```bash
# Single subject (recommended)
python src/main_benchmark_task1_task2.py \
    --task 1 --model EEGNet --strategy intra_block \
    --subject sub-01 --config config/task1_config.yaml

# All 9 sample subjects
for sub in 01 02 08 03 04 06 05 07 09; do
    python src/main_benchmark_task1_task2.py \
        --task 1 --model EEGNet --strategy intra_block \
        --subject sub-${sub} --config config/task1_config.yaml
done
```

> **Alternative:** replace `intra_block` with `intra_chrono` for a strict chronological
> 80/20 temporal split.

#### Inter-subject — 80/20 subject split

```bash
python src/main_benchmark_task1_task2.py \
    --task 1 --model EEGNet --strategy inter \
    --config config/task1_config.yaml
```

#### LOSO — Leave-One-Subject-Out

`--subject` specifies the held-out test subject; all others form the training set.

```bash
python src/main_benchmark_task1_task2.py \
    --task 1 --model EEGNet --strategy loso \
    --subject sub-01 --config config/task1_config.yaml
```

---

### Task 2 — Meditation Technique Classification

Three-class classification (HK=0 / SA=1 / BF=2) across meditation groups.

> **Sample-dataset note:** With 3 subjects per group (9 total) the group-stratified split
> falls back to a random split; AUC may appear as `nan` when the test fold contains only one
> group. This artefact is absent on the full 74-subject dataset.

#### Pre-intervention session

```bash
python src/main_benchmark_task1_task2.py \
    --task 2 --model EEGNet --strategy inter \
    --session pre --config config/task2_config.yaml
```

#### Post-intervention session

```bash
python src/main_benchmark_task1_task2.py \
    --task 2 --model EEGNet --strategy inter \
    --session post --config config/task2_config.yaml
```

---

### Task 3 — Cross-Session Adaptation

Evaluates how well a model pre-trained on pre-intervention EEG generalises to — or can be calibrated on — post-intervention EEG from the same subject.

**Dependency:** Task 3 loads a Task 1 intra-subject checkpoint. Run **Step 1** first.

#### Step 1 — Train Task 1 intra-subject checkpoints *(use `intra_block`)*

```bash
for sub in 01 02 08 03 04 06 05 07 09; do
    python src/main_benchmark_task1_task2.py \
        --task 1 --model EEGNet --strategy intra_block \
        --subject sub-${sub} --config config/task1_config.yaml
done
```

Checkpoints are saved to `results/checkpoints/task1/EEGNet/intra_block/subject_<XX>/best_*.pth`.

#### Step 2 — Zero-shot (no post-session calibration data)

```bash
python src/main_benchmark_task3.py \
    --model EEGNet --strategy zero_shot --subject sub-01 \
    --config config/task3_config.yaml \
    --task1_dir results/checkpoints/task1 --task1_intra intra_block
```

#### Step 3 — Few-shot calibration

```bash
# 5-shot
python src/main_benchmark_task3.py \
    --model EEGNet --strategy intra_5_shot --subject sub-01 \
    --config config/task3_config.yaml \
    --task1_dir results/checkpoints/task1 --task1_intra intra_block

# 10-shot
python src/main_benchmark_task3.py \
    --model EEGNet --strategy intra_10_shot --subject sub-01 \
    --config config/task3_config.yaml \
    --task1_dir results/checkpoints/task1 --task1_intra intra_block

# 30-shot
python src/main_benchmark_task3.py \
    --model EEGNet --strategy intra_30_shot --subject sub-01 \
    --config config/task3_config.yaml \
    --task1_dir results/checkpoints/task1 --task1_intra intra_block
```

---

## CLI Reference

### `main_benchmark_task1_task2.py`

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--task` | — | `1`, `2` | Benchmark task |
| `--model` | — | see [Models](#models) | Architecture |
| `--strategy` | — | `intra_block` *(recommended)*, `intra_chrono`, `inter`, `loso` | Evaluation protocol |
| `--subject` | `None` | e.g. `sub-01` | Required for `intra_*` and `loso` |
| `--session` | `pre` | `pre`, `post` | Session selector (Task 2 only) |
| `--group` | `ALL` | `ALL`, `HK`, `SA`, `BF` | Optional group filter |
| `--config` | — | path | YAML config file |
| `--epochs` | `None` | int | Override config epoch count |

### `main_benchmark_task3.py`

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--model` | — | see [Models](#models) | Architecture |
| `--strategy` | — | `zero_shot`, `intra_5_shot`, `intra_10_shot`, `intra_30_shot`, … | Adaptation protocol |
| `--subject` | — | e.g. `sub-01` | Target subject |
| `--task1_dir` | — | path | Root dir of Task 1 checkpoints |
| `--task1_intra` | `intra_block` | `intra_block` *(recommended)*, `intra_chrono` | Which Task 1 split to load weights from |
| `--ft_mode` | `full` | `full`, `linear` | Fine-tune all params or classifier head only |
| `--config` | — | path | YAML config file |
| `--ft_epochs` | `None` | int | Override fine-tuning epoch count |

---

## Models

| `--model` | Architecture | Reference |
|-----------|-------------|-----------|
| `EEGNet` | Compact depthwise-separable CNN for BCI | Lawhern et al. (2018) [1] |
| `ShallowConvNet` | Shallow temporal-spatial CNN | Schirrmeister et al. (2017) [2] |
| `DeepConvNet` | Deep temporal-spatial CNN | Schirrmeister et al. (2017) [2] |
| `Conformer` | Convolutional transformer hybrid | Song et al. (2022) [3] |
| `DGCNN` | Dynamic graph convolutional network | Song et al. (2018) [4] |

---

## Configuration

Task-specific hyperparameters live in `config/task{1,2,3}_config.yaml`. Key fields:

```yaml
experiment:
  epochs: 250          # Maximum training epochs (Task 1/2; early-stopped by patience)
  patience: 50         # Early-stopping patience (epochs without val BAcc improvement)
  batch_size: 32

data:
  data_root: "../data"            # Root directory containing derivatives/ml_preproc_data/
  tsv_path:  "../data/participants.tsv"
  num_classes: 2                  # 2 for Tasks 1 & 3 | 3 for Task 2
  num_channels: 64                # EEG channels
  sig_length: 1000                # Window samples (4 s × 250 Hz)

models:
  EEGNet:
    kernel_size: 125              # Temporal kernel (~0.5 s at 250 Hz)
    f1: 16                        # Number of temporal filters
    D: 4                          # Depth multiplier
    dropout_rate: 0.66
    lr: 0.00025
    weight_decay: 0.00005
```

To use a different data directory, update `data_root` and `tsv_path` in the config (absolute paths are accepted).

---

## Benchmark Results

> Results below are reported on the **full 74-subject dataset**. The 9-subject sample included
> in this repository is intended for pipeline verification only and will yield different values.

### Task 1 — Cognitive State Decoding (BAcc, %)

| Model | Intra-subject | Inter-subject | LOSO |
|-------|:---:|:---:|:---:|
| EEGNet | — | — | — |
| ShallowConvNet | — | — | — |
| DeepConvNet | — | — | — |
| Conformer | — | — | — |
| DGCNN | — | — | — |

### Task 2 — Technique Classification (BAcc, %)

| Model | Pre-session | Post-session |
|-------|:---:|:---:|
| EEGNet | — | — |
| ShallowConvNet | — | — |
| DeepConvNet | — | — |
| Conformer | — | — |
| DGCNN | — | — |

### Task 3 — Cross-Session Adaptation (BAcc, %)

| Model | Zero-shot | 5-shot | 10-shot | 30-shot |
|-------|:---:|:---:|:---:|:---:|
| EEGNet | — | — | — | — |
| ShallowConvNet | — | — | — | — |
| DeepConvNet | — | — | — | — |
| Conformer | — | — | — | — |
| DGCNN | — | — | — | — |

*Full results and analysis are reported in the accompanying paper.*

---

## Expected Output

Each run prints a structured result block to stdout:

```
============================================================
Run    : T1_EEGNet_intra_block_sub-sub-01
Device : cuda
Epochs : 250  |  Patience : 50
============================================================
[INFO] Initialize Dataset | Task 1 | Preprocessing: preproc | Group: ALL | Scanned files: 36
[INFO] Dataset index built: 4194 windows via RAM.
   -> Train labels: {0: 167, 1: 167}
   -> Val labels:   {0: 19, 1: 19}
   -> Test labels:  {0: 39, 1: 39}

  Epoch   1/250 | Loss: 0.7417 | Val Acc: 0.5263 | Val BAcc: 0.5263
  ...
  Early stopping at epoch 62

============================================================
RESULT  T1_EEGNet_intra_block_sub-sub-01
  Acc    : 0.7234
  BAcc   : 0.7219
  F1     : 0.7231
  AUC    : 0.7854
  PR-AUC : 0.7512
============================================================
```

Checkpoints are saved to `results/checkpoints/task{1,2,3}/`.

---

## References

[1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces," *J. Neural Eng.*, vol. 15, no. 5, p. 056013, 2018.

[2] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Hum. Brain Mapp.*, vol. 38, no. 11, pp. 5391–5420, 2017.

[3] Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG conformer: Convolutional transformer for EEG decoding and visualization," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 31, pp. 710–719, 2022.

[4] T. Song, W. Zheng, P. Song, and Z. Cui, "EEG emotion recognition using dynamical graph convolutional neural networks," *IEEE Trans. Affect. Comput.*, vol. 11, no. 3, pp. 532–541, 2018.

---

## Citation

If you use L-FAME in your research, please cite:

```bibtex
@article{lfame2025,
  title   = {L-FAME: A Longitudinal Focus Attention Meditation EEG Benchmark},
  author  = {[Author list]},
  journal = {[Venue]},
  year    = {2025}
}
```

---

## License

This sample dataset and code are released for **non-commercial academic research** under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use requires explicit written permission from the authors.
For full dataset terms, see the [Hugging Face dataset page](https://huggingface.co/datasets/L-FAME-Dataset-Benchmark/L-FAME).
