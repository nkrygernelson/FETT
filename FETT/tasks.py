"""Convenience runners for the FETT pipeline (managed via uv + invoke)."""
import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fett"


# ── Data commands ────────────────────────────────────────────────────────────

@task
def parse_sources(ctx: Context) -> None:
    """Parse raw JSON / matminer sources into per-source CSVs (interim/)."""
    ctx.run("uv run python scripts/data/01_parse_sources.py", echo=True, pty=not WINDOWS)


@task
def query_mp(ctx: Context) -> None:
    """Query Materials Project for per-functional band-gap CSVs (needs $MP_API_KEY)."""
    ctx.run("uv run python scripts/data/02_query_mp.py", echo=True, pty=not WINDOWS)


@task
def combine_raw(ctx: Context) -> None:
    """Combine per-source CSVs into per-functional raw CSVs (data/raw/homemade/)."""
    ctx.run("uv run python scripts/data/03_combine_raw.py", echo=True, pty=not WINDOWS)


@task
def make_data(ctx: Context, mode: str = "homemade") -> None:
    """Build train/val/test CSVs for a given data config (default: homemade)."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data={mode}",
        echo=True, pty=not WINDOWS,
    )


# ── Training commands ─────────────────────────────────────────────────────────

@task
def train(ctx: Context) -> None:
    """Train the multi-fidelity base bandgap model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_translation(ctx: Context, base_ckpt: str = "", data: str = "homemade_pbesol_to_all", cv: int = 1) -> None:
    """
    Train the translation head on top of a frozen base model.

    Example:
        uv run invoke train-translation --base-ckpt models/best.ckpt
        uv run invoke train-translation --base-ckpt models/best.ckpt --cv 5
    """
    overrides = []
    if base_ckpt:
        overrides.append(f"model.base_model_checkpoint={base_ckpt}")
    if data:
        overrides.append(f"data={data}")
    if cv > 1:
        overrides.append(f"training.cv_folds={cv}")
    override_str = (" " + " ".join(overrides)) if overrides else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train_translation.py{override_str}",
        echo=True, pty=not WINDOWS,
    )


# ── Evaluation commands ───────────────────────────────────────────────────────

@task
def evaluate(ctx: Context, ckpt: str = "") -> None:
    """Evaluate the base model on its test set, broken down per fidelity."""
    override = f" eval.checkpoint={ckpt}" if ckpt else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py{override}",
        echo=True, pty=not WINDOWS,
    )


@task
def evaluate_translation(ctx: Context, ckpt: str = "", data: str = "homemade_pbesol_to_all") -> None:
    """Evaluate the translation model on its test set, broken down per fidelity pair."""
    parts = ["eval.mode=translation", f"data={data}"]
    if ckpt:
        parts.append(f"eval.checkpoint={ckpt}")
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py {' '.join(parts)}",
        echo=True, pty=not WINDOWS,
    )


# ── Test commands ─────────────────────────────────────────────────────────────

@task
def test(ctx: Context) -> None:
    """Run the test suite."""
    ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)
