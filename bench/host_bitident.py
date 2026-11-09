# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Print stable per-step hashes for bit-identical host decode comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel
from host.weights import Weights


def _array_sha256(value: np.ndarray) -> str:
    """Hash metadata and exact in-memory element bytes in canonical C order."""
    arr = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(arr.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _resolve_model_dir(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    for env_name in ("RAON_MODEL_DIR", "RAON_WEIGHTS_D"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (
            (candidate / "model.safetensors").is_file()
            or (candidate / "model.safetensors.index.json").is_file()
        ):
            return candidate
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode with the NumPy host backend and hash every observable array."
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("-n", "--num-tokens", type=int, default=4)
    parser.add_argument(
        "--tokens",
        help="Comma-separated input token IDs; overrides --num-tokens.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_dir = _resolve_model_dir(args.model_dir)
    if model_dir is None:
        print(
            "host_bitident: no host checkpoint found; set RAON_MODEL_DIR or pass --model-dir",
            file=sys.stderr,
        )
        return 2
    if args.num_tokens < 1:
        print("host_bitident: --num-tokens must be positive", file=sys.stderr)
        return 2

    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = RaonConfig.from_dict(config_data)
    if args.tokens:
        try:
            token_ids = [int(part.strip()) for part in args.tokens.split(",") if part.strip()]
        except ValueError:
            print("host_bitident: --tokens must contain decimal integers", file=sys.stderr)
            return 2
        if not token_ids:
            print("host_bitident: --tokens is empty", file=sys.stderr)
            return 2
    else:
        token_ids = [index % config.vocab_size for index in range(1, args.num_tokens + 1)]
    if any(token < 0 or token >= config.vocab_size for token in token_ids):
        print(f"host_bitident: token outside [0, {config.vocab_size})", file=sys.stderr)
        return 2

    weights = Weights(model_dir)
    model = RaonModel(config, weights, backend=NumpyBackend(weights))
    print(
        json.dumps(
            {
                "model_dir": str(model_dir.resolve()),
                "tokens": token_ids,
                "bf16_storage": "uint16",
            },
            sort_keys=True,
        )
    )
    for step, token_id in enumerate(token_ids, start=1):
        logits = model.step(token_id)
        print(f"step={step} token={token_id} logits sha256={_array_sha256(logits)}")
        for name in sorted(model.last_intermediates):
            value = model.last_intermediates[name]
            print(f"step={step} intermediate={name} sha256={_array_sha256(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
