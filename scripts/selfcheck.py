# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed greedy installation check on the NPU."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

EXPECTED_HASH = "3dc9191d4d3d6344"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kernel_object_audit import audit
from scripts.selfcheck_prompt import messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-d', type=Path, default=os.environ.get('RAON_WEIGHTS_D'))
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    if args.weights_d is None:
        parser.error('provide --weights-d or set RAON_WEIGHTS_D')
    if not args.weights_d.is_dir():
        parser.error('--weights-d must be an existing directory')
    if audit(REPO)['stale']:
        print('Run scripts\\run.cmd scripts\\kernel_object_audit.py --fix first.')
        return 1
    if not os.environ.get('RAON_PREFILL_CHUNK'):
        os.environ['RAON_PREFILL_CHUNK'] = '256'
        print('RAON_PREFILL_CHUNK was empty; set to 256.')
    else:
        print('RAON_PREFILL_CHUNK=' + os.environ['RAON_PREFILL_CHUNK'])
    print('First run builds the NPU kernels (64 xclbin files; about 4-5 minutes on the test laptop).', flush=True)
    from serve.engine import Engine, GenOptions
    started = time.perf_counter()
    eng = Engine(args.weights_d, args.model_dir or args.weights_d, preload=True)
    load_s = time.perf_counter() - started
    chunks = list(eng.chat(messages, options=GenOptions(temperature=0, max_tokens=32)))
    text = ''.join(c.text for c in chunks)
    thinking = ''.join(c.thinking for c in chunks)
    actual = hashlib.sha256((thinking + '|' + text).encode()).hexdigest()[:16]
    print(f'load_s={load_s:.3f}, hash={actual}')
    if chunks:
        last = chunks[-1]
        print(f'prefill_s={last.prefill_s}, decode_s={last.decode_s}, completion_tokens={last.completion_tokens}')
    passed = bool(chunks) and actual == EXPECTED_HASH
    print('RESULT: PASS' if passed else 'RESULT: FAIL')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
