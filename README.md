# Hierarchical FL with Bidirectional Low-Rank Compression & Coordinated Feature Selection

Reference implementation for:
> *Communication-Efficient Hierarchical Federated Learning via Model-Guided
> Entropy–Correlation Feature Selection and Bidirectional Low-Rank Compression.*

Archived release: [Zenodo DOI 10.5281/zenodo.21729792](https://doi.org/10.5281/zenodo.21729792).
Upload this corrected version as a new version of the same Zenodo record before
the revised manuscript is resubmitted.

This repository reproduces **all 13 methods**, the **3-tier hierarchical**
protocol, the **3-link communication accounting**, the **coordinated feature
mask** (UCI HAR), and the **CIFAR-10 matched protocol (100 clients / 400
rounds)** reported in the paper. Trajectory figures are plotted from the
**real per-round CSV/JSON logs** produced by `run.py` (no synthetic curves).

## Methods implemented
`fedavg`, `fedprox`, `scaffold`, `fednova`, `fedkd` (disclosed adaptation),
`qsgd`, `fedpaq`, `fedcom`, `topk` (Top-k + error feedback),
`adaptive_topk`, `sign` (SignSGD majority vote), `powersgd`, `proposed`.

## Install
Python 3.10 or newer is required.
```bash
git clone <repository-url> fl-hiersvd && cd fl-hiersvd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data
- **Fashion-MNIST / CIFAR-10**: auto-downloaded by `torchvision`.
- **UCI HAR**: download `UCI HAR Dataset.zip` from the
  [UCI repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones),
  unzip so that `data/HAR/train/X_train.txt`, `train/y_train.txt`,
  `test/X_test.txt`, `test/y_test.txt` exist.

## Exact experimental settings

### Main protocol — Table 2 (Fashion-MNIST & UCI HAR)
| Parameter | Value |
|---|---|
| Clients / Edges | 50 / 5 (10 clients/edge) |
| Rounds | 30 |
| Local epochs / batch / lr | 5 / 64 / 0.05 (SGD) |
| SVD rank *k* | 10 |
| Dirichlet α | 0.1 (non-IID) |
| Seeds | 42, 43, 44, 45, 46 |
| Feature selection (HAR only) | warm-up *T_w*=5, ρ=0.70, weights (0.60, 0.25, 0.15) |
| Tabular model | MLP, one hidden layer with 128 units |

### Baseline-specific (Table 3)
| Method | Setting | Counted communication |
|---|---|---|
| FedProx | μ=0.01 | dense, 3 links |
| SCAFFOLD | option-II control update | **model + control variate**, 3 links |
| FedNova | minibatch-step normalisation | dense, 3 links |
| FedKD | teacher 256 / student 128 / λ=0.3 / T=1 / rank 10 | dense student downlink, SVD student uplink, dense edge agg |
| PowerSGD-EF | rank 10, warm start, client residuals | P/Q uplinks, edge factors, dense global downlink |
| QSGD-8 | 255 levels | norm + 8-bit levels + sign |
| FedPAQ-8 | QSGD-8 + 25/50 sampled clients | selected-client links + active-edge agg |
| FedCOM-8 | QSGD-8 + γ=1.5 | quantized uplink, dense others |
| Top-k-EF | 10% + client residuals | float values + uint32 indices |
| Adaptive-TopK-EF | 99% energy, bounded 1–20% | float values + uint32 indices |
| SignSGD-MV | hierarchical majority vote | 1 bit/coord, 3 links |
| Proposed | rank 10 + downlink error feedback | SVD factors on all 3 links |

### CIFAR-10 matched protocol — Table 7
| Parameter | Value |
|---|---|
| Clients / Edges | **100 / 5** |
| Rounds | **400** |
| Local epochs / batch / lr | 5 / 64 / 0.05 |
| SVD rank *k* | 10 |
| Methods | FedAvg, PowerSGD-EF, QSGD-8, Proposed |
| CNN | Conv32→Conv64→Pool→FC256→FC10 (exact arch of reported numbers) |

> The CIFAR protocol differs from Table 2 *only* in clients/rounds because a
> 4-layer CNN needs more rounds to converge under α=0.1; all other
> hyper-parameters follow Table 2.

## Run
Single run:
```bash
python run.py --method proposed --dataset har --seed 42
```
Everything (all methods × datasets × seeds + CIFAR):
```bash
bash scripts/run_all.sh
```
Build tables / figures from the logs:
```bash
python scripts/make_tables.py
python scripts/make_figures.py
```
Run the separate fresh-process timing and scalability protocols:
```bash
bash scripts/run_profile.sh
bash scripts/run_scalability.sh
```
Re-run the training-only HAR retention sweep (the official test set is not
loaded by this script):
```bash
python scripts/run_fs_validation.py
```

## Output layout
`results/{dataset}_{method}_s{seed}.json` contains the full per-round
trajectory (`acc`, `f1`, `comm_mb`) and the final 3-link breakdown
(`c2e`, `e2c`, `down`, `setup`), so every reported number is reproducible.
As in the article's numerical tables, the `MB` label denotes MiB
(`bytes / 1024²`) for continuity with the archived experiment logs.

## Validation
Run the fast unit and synthetic end-to-end tests before launching the full grid:
```bash
pytest -q
```

## Notes on the figures
- Fig 1 (architecture) and the Algorithm are rendered in LaTeX (TikZ /
  `algorithmic`), not from this repo.
- `scripts/make_figures.py` creates the two-dataset accuracy and Macro-F1
  trajectories, the final communication bars, the HAR mask ablation, and both
  CIFAR-10 figures directly from per-seed logs.
- The full experiment grid is computationally expensive. Existing article
  numbers should be replaced only after all five seeds complete successfully;
  smoke tests establish code-path integrity, not numerical reproduction.

## License
MIT (see `LICENSE`).
