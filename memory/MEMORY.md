# FETT Project Memory

## Project Overview
Multi-fidelity band gap prediction using a Fidelity Embedding Transformer Translation (FETT).
Fidelity levels: PBE/GGA (0), SCAN (1), GLLB-SC (2), HSE (3), EXPT (4).

## Key Paths
- Config: `FETT/configs/` — Hydra configs for data, model, training, eval
- Raw data: `FETT/data/raw/homemade/{pbe,SCAN,GLLBSC,HSE,EXPT}.csv`
- Source data: `FETT/data/source/` — per-source per-functional CSVs
- JSON sources: `FETT/data/raw/homemade/source/` — large JSON archives
- Scripts: `FETT/scripts/data/` — one-time data prep scripts

## Data Pipeline (complete)
Three one-time scripts in `FETT/scripts/data/`:
1. `01_parse_sources.py` — JSON + matminer → `data/source/` CSVs
2. `02_query_mp.py` — MP API → `data/source/mp_*.csv` (needs MP_API_KEY, uses chunk_size=1000)
3. `03_combine_raw.py` — source CSVs → `data/raw/homemade/*.csv`

Run order: 01 → 02 → 03, then `uv run invoke make-data`

## Dataset Modes (make_dataset.py, switchable via Hydra data: config)
- `homemade` (standard): all fidelities, standard 60/20/20 split
- `homemade_only_new_on_expt`: EXPT test = formulas NOT in any lower-fidelity training set
- `homemade_translation`: matched pairs (source_fidelity→target_fidelity) for translation model

## Model Architecture
- `SetBasedBandgapModel`: set-based deep learning, fidelity-conditioned, 5 pooling types
  - `forward(..., return_embedding=True)` returns pooled embedding [batch, 192] before prediction head
- `FettLightningModule`: Lightning wrapper, embeds model_cfg in every checkpoint (on_save_checkpoint)
- `FettTranslationModule`: frozen base model + trainable `FidelityTranslationHead`
- `FidelityTranslationHead`: MLP(base_emb + target_fid_emb + source_bg → target_bg)
- `model_io.load_model_from_checkpoint(path)`: reconstruct model from embedded model_cfg in .ckpt

## Training
- Base model: `uv run invoke train`
- Translation model: `uv run invoke train-translation --base-ckpt models/best.ckpt`
- Checkpoints saved to `models/` with embedded model_cfg for future reconstruction
- EarlyStopping(patience=15), ModelCheckpoint(val/loss), gradient_clip=1.0

## Evaluation
- Base model: `uv run invoke evaluate --ckpt models/best.ckpt`
- Translation model: `uv run invoke evaluate-translation --ckpt models/best_translation.ckpt`
- Outputs: per-fidelity scatter plots + summary bar chart + CSVs in `reports/figures/`

## Tech Stack
- Python 3.13 (requires >=3.12), uv package manager
- Hydra (config), Lightning (training), PyTorch CPU build
- pymatgen, mp-api, matminer, matplotlib (data processing + eval)
- Package structure: `FETT/src/fett/` with model/, data/ subpackages (all have __init__.py)

## Config Paths (Hydra config_path)
- Scripts in `src/fett/` (train.py, evaluate.py): `config_path="../../configs"`
- Scripts in `src/fett/data/` (make_dataset.py): `config_path="../../../configs"`

## Invoke Tasks
- make-data, make-data-only-new, make-data-translation
- train, train-translation
- evaluate, evaluate-translation
- test, docker-build, build-docs, serve-docs
