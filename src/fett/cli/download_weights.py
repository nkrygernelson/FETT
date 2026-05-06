"""CLI entry point for downloading FETT model weights from the Hugging Face Hub."""
import argparse

from fett.model.weights import DEFAULT_REPO_ID, download_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Download FETT weights from the Hugging Face Hub.")
    p.add_argument("--kind", default="base", help="Checkpoint kind: 'base', 'translation', or a raw filename.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF Hub repo id, e.g. user/name.")
    p.add_argument("--out", default="models", help="Local directory to save the file in.")
    p.add_argument("--revision", default=None, help="Optional git revision (branch/tag/commit) on the Hub.")
    p.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = p.parse_args()

    path = download_checkpoint(
        kind=args.kind,
        repo_id=args.repo_id,
        local_dir=args.out,
        revision=args.revision,
        force=args.force,
    )
    print(f"Downloaded {args.kind} → {path}")


if __name__ == "__main__":
    main()
