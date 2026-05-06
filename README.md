# FETT — Fidelity-Embedding Transformer Translator

Reference implementation of the multi-fidelity bandgap predictor and the cross-fidelity
translation pipeline (e.g. PBEsol → experimental).

The pipeline has two models:

1. **Base model** — a permutation-invariant transformer over element embeddings, trained on
   ~7 fidelity levels (PBE, PBE+U, PBEsol, SCAN, GLLB-SC, HSE, experimental). Predicts
   bandgaps from composition + fidelity label and exposes a pooled compound embedding.
2. **Translation head** — an MLP that takes two pooled embeddings (source-fidelity and
   target-fidelity) plus the source bandgap and predicts the target-fidelity bandgap.

## Setup

```bash
uv sync                  # core deps (training / eval)
uv sync --extra data     # also install matminer + mp-api for rebuilding raw data
uv sync --extra wandb    # also install wandb for optional W&B logging
```

Enable W&B logging with a Hydra override:
```bash
uv run invoke train -- training.logger.wandb.enabled=true training.logger.wandb.project=fett
```

## Pretrained weights

Released checkpoints are mirrored on the Hugging Face Hub
(default repo: `nkrygernelson/fett-bandgap`). Override with the `FETT_HF_REPO`
environment variable or the `--repo-id` flag.

```bash
# Download into ./models/
uv run invoke download-weights                       # base.ckpt
uv run invoke download-weights --kind translation    # translation head

# Then evaluate without training:
uv run invoke evaluate --ckpt models/best.ckpt
uv run invoke evaluate-translation --ckpt models/best_translation.ckpt
```

In Python:
```python
from fett.model.weights import download_checkpoint
ckpt_path = download_checkpoint("base")
```

To publish a freshly trained checkpoint (requires `HF_TOKEN`):
```bash
uv run invoke publish-weights --ckpt models/best.ckpt
uv run invoke publish-weights --ckpt models/best_translation.ckpt --kind translation
```

## Workflow

```bash
# 1. (optional) rebuild raw CSVs from public sources
export MP_API_KEY=<your-key>
uv run invoke parse-sources
uv run invoke query-mp
uv run invoke combine-raw

# 2. Build train/val/test splits for the base model
uv run invoke make-data                            # data=homemade

# 3. Train the base model — saves to models/best.ckpt
uv run invoke train

# 4. Evaluate the base model per fidelity (writes plots + CSVs to reports/figures/)
uv run invoke evaluate --ckpt models/best.ckpt

# 5. Build a translation dataset (e.g. PBEsol → all higher fidelities)
uv run invoke make-data --mode homemade_pbesol_to_all

# 6. Train the translation head on top of the frozen base model
uv run invoke train-translation --base-ckpt models/best.ckpt
# 5-fold cross-validation (paper Table 4):
uv run invoke train-translation --base-ckpt models/best.ckpt --cv 5

# 7. Evaluate the translation model per fidelity pair
uv run invoke evaluate-translation --ckpt models/best_translation.ckpt
```

Other dataset modes available via `--mode`: `homemade`, `homemade_only_new_on_expt`,
`homemade_translation`, `homemade_pbe_to_all`, `homemade_pbesol_to_all`,
`homemade_to_expt`. Hydra overrides work for any flag, e.g.
`uv run src/fett/train.py training.batch_size=128`.

## Layout

```
configs/      Hydra configs (data, model, training, eval)
scripts/data/ Raw-data ingestion (sources → interim → raw)
src/fett/
  data/       Dataset loaders + make_dataset.py
  model/      SetBasedBandgapModel, attention pooling, translation head, Lightning modules
  train.py            Train base
  train_translation.py Train translation head (single split or K-fold CV)
  evaluate.py         Per-fidelity / per-pair metrics + parity plots
  cli/        download_weights / publish_weights CLIs
  model/weights.py    Hugging Face Hub download + publish helpers
data/         raw/, interim/, processed/ (gitignored)
models/       checkpoints (gitignored — fetched via `invoke download-weights`)
reports/      figures + per-split predictions (gitignored)
```

## License

MIT — see `LICENSE`.
