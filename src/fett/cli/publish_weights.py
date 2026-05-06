"""CLI entry point for publishing FETT model weights to the Hugging Face Hub."""
import argparse

from fett.model.weights import DEFAULT_REPO_ID, publish_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Publish a FETT checkpoint to the Hugging Face Hub.")
    p.add_argument("--ckpt", required=True, help="Local path to the .ckpt file.")
    p.add_argument("--kind", default="base", help="Checkpoint kind: 'base', 'translation', or a raw filename.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF Hub repo id, e.g. user/name.")
    p.add_argument("--private", action="store_true", help="Create the repo as private (only on first upload).")
    p.add_argument("--message", default=None, help="Optional commit message.")
    args = p.parse_args()

    url = publish_checkpoint(
        args.ckpt,
        kind=args.kind,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.message,
    )
    print(f"Uploaded → {url}")


if __name__ == "__main__":
    main()
