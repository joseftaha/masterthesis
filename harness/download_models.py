"""One-shot helper that pre-fetches the two model checkpoints used by
the experiments into ``$HF_HOME``. Run on a login node with internet
access *before* submitting any sbatch jobs so that compute nodes never
exercise the remote download path.

Re-runs are idempotent (HF cache).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise SystemExit("HF_HOME must be set (point at local NVMe scratch)")
    print(f"HF_HOME = {hf_home}")
    for side in ("primary", "secondary"):
        m = cfg["models"][side]
        repo = m.get("quant_repo_override") or m["hf_repo"]
        print(f"-> {repo}")
        snapshot_download(repo_id=repo, revision=m.get("revision") or "main",
                          local_files_only=False)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
