"""
Download and publish trained FETT model weights from/to the Hugging Face Hub.

Researchers reproducing the paper can fetch the released checkpoints with::

    from fett.model.weights import download_checkpoint
    ckpt_path = download_checkpoint("base")        # base bandgap model
    ckpt_path = download_checkpoint("translation") # translation head

The author can publish a freshly trained checkpoint with::

    uv run invoke publish-weights --kind base --ckpt models/best.ckpt

The default repo is configured below; override with the ``repo_id`` argument or
the ``FETT_HF_REPO`` environment variable. Authentication for upload uses the
standard ``HF_TOKEN`` environment variable (or ``huggingface-cli login``).
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get("FETT_HF_REPO", "nkrygernelson/fett-bandgap")

# Canonical filenames for each released checkpoint.
CHECKPOINTS: dict[str, str] = {
    "base": "best.ckpt",
    "translation": "best_translation.ckpt",
}


def download_checkpoint(
    kind: str = "base",
    *,
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: str | Path = "models",
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """
    Download a released checkpoint from the Hugging Face Hub.

    Args:
        kind: Which checkpoint to fetch. One of ``CHECKPOINTS`` keys
            (``"base"`` or ``"translation"``), or a raw filename in the repo.
        repo_id: HF Hub repo, ``"<user>/<name>"``. Defaults to ``FETT_HF_REPO``
            env var or the project default.
        local_dir: Directory to place the file in (created if missing).
        revision: Optional git revision (branch / tag / commit) on the Hub.
        force: Re-download even if a cached copy exists.

    Returns:
        Local path to the downloaded checkpoint.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download weights. "
            "Install with `uv sync` (it's a core dependency)."
        ) from exc

    filename = CHECKPOINTS.get(kind, kind)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
        force_download=force,
    )
    return Path(path)


def publish_checkpoint(
    ckpt_path: str | Path,
    *,
    kind: str = "base",
    repo_id: str = DEFAULT_REPO_ID,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """
    Upload a checkpoint to the Hugging Face Hub. Creates the repo if needed.

    Requires authentication via ``HF_TOKEN`` env var or ``huggingface-cli login``.

    Args:
        ckpt_path: Local path to the ``.ckpt`` file to upload.
        kind: Logical name (``"base"`` / ``"translation"``); selects the
            destination filename from ``CHECKPOINTS``. Pass an arbitrary string
            to upload under a custom filename.
        repo_id: HF Hub repo, ``"<user>/<name>"``.
        private: Create the repo as private if it does not yet exist.
        commit_message: Optional commit message for the upload.

    Returns:
        URL of the uploaded file on the Hub.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to publish weights. "
            "Install with `uv sync`."
        ) from exc

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    filename = CHECKPOINTS.get(kind, kind)
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    api = HfApi()
    return api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or f"Upload {kind} checkpoint",
    )
