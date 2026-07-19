"""Credential-free command line entry point for upload planning."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from core import UploadJob, UploadResult
from uploading.ledger import JsonUploadLedger
from uploading.service import UploadService
from uploading.youtube import YouTubeUploadAdapter


def main(argv: Sequence[str] | None = None) -> int:
    """Print one deterministic YouTube dry-run plan as JSON."""

    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("This Prototype 1 entry point requires --dry-run.")
    job = UploadJob(
        rendered_clip_path=args.clip,
        rendered_clip_identity=args.render_identity,
        destination="youtube",
        title=args.title,
        description=args.description,
        tags=args.tag,
        visibility=args.privacy_status,
    )
    service = UploadService(
        JsonUploadLedger(args.ledger),
        [YouTubeUploadAdapter()],
    )
    result = service.execute(job, dry_run=True)
    print(json.dumps(_result_value(result), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a credential-free YouTube upload for a rendered clip."
    )
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--render-identity", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--privacy-status",
        choices=("private", "unlisted", "public"),
        default="private",
    )
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("data") / "uploads" / "upload-ledger.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _result_value(result: UploadResult) -> dict[str, object]:
    return {
        "upload_identity": result.upload_identity,
        "rendered_clip_identity": result.rendered_clip_identity,
        "rendered_clip_path": str(result.rendered_clip_path),
        "destination": result.destination,
        "status": result.status.value,
        "remote_id": result.remote_id,
        "remote_url": result.remote_url,
        "recovered": result.recovered,
        "metadata": result.metadata,
    }
