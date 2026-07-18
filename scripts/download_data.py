#!/usr/bin/env python3
"""Download and verify the frozen DANDI NWB assets used by the analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_TEMPLATE = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(asset: dict, output_root: Path, retries: int = 4) -> tuple[str, str]:
    target = output_root / asset["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(asset["size"])
    expected_sha = asset["digest"]["dandi:sha2-256"]

    if target.exists() and target.stat().st_size == expected_size:
        if sha256(target) == expected_sha:
            return asset["path"], "verified-existing"

    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        partial.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                API_TEMPLATE.format(asset_id=asset["asset_id"]),
                headers={"User-Agent": "active-touch-reproduction/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch: {partial.stat().st_size} != {expected_size}"
                )
            if sha256(partial) != expected_sha:
                raise RuntimeError("SHA-256 mismatch")
            os.replace(partial, target)
            return asset["path"], "downloaded-verified"
        except Exception as error:  # noqa: BLE001 - retry network and integrity failures
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{asset['path']}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    assets = manifest["results"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, asset, args.output_root): asset for asset in assets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            asset = futures[future]
            try:
                path, status = future.result()
                print(f"[{completed:02d}/{len(assets):02d}] {status}: {path}", flush=True)
            except Exception as error:  # noqa: BLE001 - collect all failed assets
                message = f"{asset['path']}: {error}"
                failures.append(message)
                print(f"[{completed:02d}/{len(assets):02d}] FAILED: {message}", flush=True)

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
