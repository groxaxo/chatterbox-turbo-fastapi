#!/usr/bin/env python3
"""Download the official base model and selected Lucía artifacts for offline use."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


ALL_PROFILES = ("lucia-ar", "lucia-latam", "lucia-cl-pilot", "lucia-co-pilot")
CONTINUAL_LORA_PROFILES = {"lucia-latam", "lucia-cl-pilot", "lucia-co-pilot"}
PILOT_PROFILES = {"lucia-cl-pilot", "lucia-co-pilot"}


def expand_profile_dependencies(requested: list[str]) -> list[str]:
    """Include the artifacts needed to reconstruct each profile's exact base chain."""
    expanded = list(dict.fromkeys(requested))
    if any(profile in CONTINUAL_LORA_PROFILES for profile in expanded):
        expanded.insert(0, "lucia-ar")
    if any(profile in PILOT_PROFILES for profile in expanded) and "lucia-latam" not in expanded:
        # Pilot directories contain adapters only; persona conditioning is inherited
        # from the released lucia-latam profile.
        insert_at = 1 if expanded and expanded[0] == "lucia-ar" else 0
        expanded.insert(insert_at, "lucia-latam")
    return list(dict.fromkeys(expanded))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models", help="Parent directory for downloaded models")
    parser.add_argument("--base-repo", default="ResembleAI/chatterbox-turbo")
    parser.add_argument("--spanish-repo", default="groxaxo/chaturbo-espanol")
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--spanish-revision", default="main")
    parser.add_argument(
        "--profiles",
        default="lucia-ar,lucia-latam",
        help="Comma-separated profiles or 'all'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    base_dir = root / "chatterbox-turbo"
    spanish_dir = root / "chaturbo-espanol"
    token = os.getenv("HF_TOKEN") or None

    requested = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if requested == ["all"]:
        requested = list(ALL_PROFILES)
    unknown = sorted(set(requested).difference(ALL_PROFILES))
    if unknown:
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown)}")
    download_profiles = expand_profile_dependencies(requested)

    base_dir.mkdir(parents=True, exist_ok=True)
    spanish_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.base_repo}@{args.base_revision} -> {base_dir}")
    snapshot_download(
        repo_id=args.base_repo,
        revision=args.base_revision,
        local_dir=base_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        token=token,
    )

    print(
        f"Downloading {args.spanish_repo}@{args.spanish_revision} -> {spanish_dir} "
        f"(requested={requested}, dependency-closed={download_profiles})"
    )
    snapshot_download(
        repo_id=args.spanish_repo,
        revision=args.spanish_revision,
        local_dir=spanish_dir,
        allow_patterns=[f"{profile}/**" for profile in download_profiles],
        token=token,
    )

    print("\nUse these environment variables:")
    print(f"BASE_MODEL_DIR={base_dir}")
    print(f"SPANISH_MODEL_DIR={spanish_dir}")
    print(f"DEFAULT_SPANISH_PROFILE={requested[0] if requested else 'lucia-ar'}")


if __name__ == "__main__":
    main()
