#!/usr/bin/env python3
"""Verify that the inherited SemiCat repository remains untouched.

The BCFM implementation is additive: inherited tracked files must match the
canonical upstream checkout byte-for-byte. Only repository hygiene/dependency
metadata are deliberately exempted, and their names are reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_UPSTREAM_COMMIT = "558602a0fa722514e4a6012f5c46a8ae178b3068"
ALLOWED_INFRASTRUCTURE_DIFFERENCES = {".gitignore", "environment.yaml"}
CRITICAL_HASHES = {
    "semicat/data/text8.py": "12ac54e1f1b7695126188ae7fab843c2afbc95bcdffc555b25550bfb43910a5e",
    "semicat/models/semicat.py": "82f4bf2cab2fcfa78ccfc59d9094e6c3132f5c2933c1cd7e61ff0b644ac71425",
    "semicat/models/textsemicat.py": "6cfb051b22fd10b27764df3f9058d0ec85ef00af206b6aac8753afd1571ff2c0",
    "semicat/net/duo.py": "b830eeeec56b798ad9ebcca4b18d834c5e9d52a15edcd2b14eec8dd85a13a9ff",
    "semicat/metric/text_dist.py": "ea5cf60bf3b01c69e30594a957cbcb0d7f16cffca22d6c51fc5919cb0483caec",
}


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args],
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream",
        type=Path,
        default=repo.parents[1] / "external" / "semicat_clean_20260709",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    upstream = args.upstream.resolve()

    if not (upstream / ".git").exists():
        raise FileNotFoundError(f"untouched upstream checkout not found: {upstream}")
    commit = _git(upstream, "rev-parse", "HEAD")
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"wrong upstream commit: expected {EXPECTED_UPSTREAM_COMMIT}, got {commit}"
        )

    tracked = _git(upstream, "ls-files").splitlines()
    exact: list[str] = []
    allowed: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    for relative in tracked:
        local_path = repo / relative
        upstream_path = upstream / relative
        if not local_path.is_file():
            missing.append(relative)
        elif local_path.read_bytes() == upstream_path.read_bytes():
            exact.append(relative)
        elif relative in ALLOWED_INFRASTRUCTURE_DIFFERENCES:
            allowed.append(relative)
        else:
            mismatched.append(relative)

    critical = {name: _sha256(repo / name) for name in CRITICAL_HASHES}
    bad_hashes = {
        name: {"expected": CRITICAL_HASHES[name], "actual": actual}
        for name, actual in critical.items()
        if actual != CRITICAL_HASHES[name]
    }
    report = {
        "upstream_checkout": str(upstream),
        "upstream_commit": commit,
        "tracked_files": len(tracked),
        "byte_exact_files": len(exact),
        "allowed_infrastructure_differences": sorted(allowed),
        "unexpected_mismatches": mismatched,
        "missing_files": missing,
        "critical_sha256": critical,
        "status": "pass" if not (mismatched or missing or bad_hashes) else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[parity] wrote {args.json_out.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
