import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fett"
PYTHON_VERSION = "3.12"


# ── Data commands ────────────────────────────────────────────────────────────

@task
def make_data(ctx: Context) -> None:
    """Make standard multi-fidelity dataset (data: homemade)."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data/make_dataset.py", echo=True, pty=not WINDOWS)


@task
def make_data_only_new(ctx: Context) -> None:
    """Make only_new_on_expt dataset (EXPT test has no lower-fidelity formula overlap)."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data=homemade_only_new_on_expt",
        echo=True, pty=not WINDOWS,
    )


@task
def make_data_translation(ctx: Context) -> None:
    """Make matched-pairs translation dataset — all fidelity pairs."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data=homemade_translation",
        echo=True, pty=not WINDOWS,
    )


@task
def make_data_pbe_to_all(ctx: Context) -> None:
    """Make translation dataset with plain PBE/GGA (fidelity 0) as the only source."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data=homemade_pbe_to_all",
        echo=True, pty=not WINDOWS,
    )


@task
def make_data_pbesol_to_all(ctx: Context) -> None:
    """Make translation dataset with PBEsol (fidelity 1) as the only source — primary translation task."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data=homemade_pbesol_to_all",
        echo=True, pty=not WINDOWS,
    )


@task
def make_data_to_expt(ctx: Context) -> None:
    """Make translation dataset where the target is always EXPT (fidelity 5)."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data/make_dataset.py data=homemade_to_expt",
        echo=True, pty=not WINDOWS,
    )


# ── Training commands ─────────────────────────────────────────────────────────

@task
def train(ctx: Context) -> None:
    """Train the base multi-fidelity model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_translation(ctx: Context, base_ckpt: str = "") -> None:
    """
    Train the translation head on top of a frozen base model.

    Requires a trained base checkpoint:
        uv run invoke train-translation --base-ckpt models/best.ckpt
    """
    override = f" model.base_model_checkpoint={base_ckpt}" if base_ckpt else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train_translation.py{override}",
        echo=True, pty=not WINDOWS,
    )


# ── Evaluation commands ───────────────────────────────────────────────────────

@task
def evaluate(ctx: Context, ckpt: str = "") -> None:
    """
    Evaluate the base model on the test set (per fidelity).

    Example:
        uv run invoke evaluate --ckpt models/best.ckpt
    """
    override = f" eval.checkpoint={ckpt}" if ckpt else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py{override}",
        echo=True, pty=not WINDOWS,
    )


@task
def evaluate_translation(ctx: Context, ckpt: str = "") -> None:
    """
    Evaluate the translation model on the translation test set (per fidelity pair).

    Example:
        uv run invoke evaluate-translation --ckpt models/best_translation.ckpt
    """
    override = f" eval.checkpoint={ckpt} eval.mode=translation" if ckpt else " eval.mode=translation"
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py data=homemade_translation{override}",
        echo=True, pty=not WINDOWS,
    )


# ── Test / quality commands ───────────────────────────────────────────────────

@task
def test(ctx: Context) -> None:
    """Run tests with coverage."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


# ── Docker commands ───────────────────────────────────────────────────────────

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True, pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True, pty=not WINDOWS,
    )


# ── Documentation commands ────────────────────────────────────────────────────

@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
