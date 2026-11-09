# raon-npu

[한국어](README.ko.md)

Run KRAFTON Raon-1-mini-preview (30B-A4B MoE) with NPU-only inference on AMD Ryzen AI XDNA 2, with Ollama- and OpenAI-compatible servers. This is an unofficial personal project, unaffiliated with KRAFTON or AMD. Converted weights are not distributed here.

## Status and measurements

Measured on **AMD Ryzen AI 7 350 (XDNA 2), 64 GB RAM, Windows 11 Home build 26300**, using NPU driver 32.0.203.314, XRT SDK 2.21.75, mlir-aie v1.4.4, llvm-aie 22.0.0.2026092101, IRON b3ba3cf3 and Python 3.13.15. These are observations from the 2026-11-09 engine, not performance guarantees.

| Measurement | Result |
|---|---|
| Prefill, 445-token prompt | about 14.5 s (about 31 tokens/s) |
| Decode | about 7 tokens/s (130–145 ms/token) |
| Longest prompt tested | about 6,700 tokens |
| Loaded memory | about 25 GB |
| First request weight loading | about 20–36 s |
| Reload after idle | about 16–20 s |
| Converted bundle | 26,131,993,360 bytes |
| First kernel build time | about 4–5 minutes (first selfcheck in a fresh checkout: 257 s; later runs: about 1 minute) |

## Requirements

Windows 11, AMD Ryzen AI 300-series XDNA 2 NPU and **64 GB RAM** are the target. Only the configuration above is tested. Other XDNA 2 chips, 32 GB RAM, and Linux are unverified. XDNA 1 (Ryzen 7040/8040) is unsupported.

Before setup, install:

- Git for Windows: `winget install Git.Git` (reopen PowerShell afterwards).
- Python **3.13 64-bit from [python.org](https://www.python.org/downloads/windows/)**, not Microsoft Store; enable PATH during installation.
- The NPU driver: Device Manager should show **NPU Compute Accelerator Device**. Testing used driver 32.0.203.314; use that version or newer, with newer versions requiring validation.
- Administrator access for toolchain setup. Normal serving runs as the current user.

Allow disk space for **32.7 GB safetensors + 19.7 GB GGUF** on the conversion machine, plus **26.1 GB converted output**, toolchains and temporary files. On the laptop, allow **26.1 GB weights + about 0.5 GB build files** (about 7,500 files, 64 xclbins), plus toolchains. Keep additional free space for downloads and extraction.

| Environment | Python | Invocation / dependencies |
|---|---|---|
| NPU engine | 3.13, mlir-aie IRON environment | `scripts\run.cmd`; `requirements-npu.txt` |
| Conversion | separate Python 3.12 virtual environment or another machine | `python`; `requirements-convert.txt` |

Conversion **does not need IRON**: its imports use the CPU implementations in `quant/`. The pinned conversion versions were used with Python 3.12.2 and have no Python 3.13 wheels. Do not install them into the NPU environment. The tested engine environment contains numpy 2.5.4, ml_dtypes 0.6.0, mlir-aie 1.4.4, llvm-aie 22.0.0.2026092101+0006955e, tokenizers 0.22.2 and Jinja2 3.1.6; `iron_setup.py` supplies the first four.

| Weights directory interface | Name |
|---|---|
| Server and selfcheck CLI | `--weights-d` |
| PowerShell scripts | `-WeightsD` |
| Environment | `RAON_WEIGHTS_D` |

## Installation: five steps

### 1. Clone this repository

```powershell
git clone https://github.com/whguswka/raon-npu.git raon-npu
cd raon-npu
```

### 2. Install the toolchain

From the cloned repository, in **Administrator PowerShell**:

```powershell
.\scripts\setup-toolchain.ps1
# Optional custom installation root:
# .\scripts\setup-toolchain.ps1 -Root D:\NpuTools
```

Windows may block PowerShell scripts by default. If you see a "running scripts is disabled" error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or invoke each script as `powershell -ExecutionPolicy Bypass -File <script>`.

This installs/checks Visual Studio Build Tools, XRT 2.21.75 (with SHA256 verification), mlir-aie v1.4.4, and IRON commit `b3ba3cf3a138c21e021d5ee2d6794573cd7a3287`; installs `requirements-npu.txt` inside the IRON environment; and runs SAXPY. Defaults: `C:\dev\mlir-aie`, `C:\dev\IRON`, `C:\Xilinx\XRT`. The transcript is `<Root>\raon-setup.log`. Existing mismatched versions stop setup rather than silently switching checkouts. When XRT has no version file, setup reports that the version could not be verified and checks `pyxrt` presence only.

For a custom root, set `RAON_MLIR_AIE_DIR` and `RAON_IRON_DIR` in each runtime shell (and persist them as user environment variables before creating a logon task).

### 3. Download and convert weights

Use a **separate Python 3.12 environment**. First accept the model's access terms on the [model page](https://huggingface.co/KRAFTON/Raon-1-mini-preview) with a Hugging Face account; also check access to the [GGUF repository](https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf). Then authenticate:

```powershell
py -3.12 -m venv .venv-convert
.\.venv-convert\Scripts\Activate.ps1
python -m pip install -r requirements-convert.txt
hf auth login
hf download KRAFTON/Raon-1-mini-preview --local-dir hf/Raon-1-mini-preview
hf download KRAFTON/Raon-1-mini-preview-gguf raon-1-mini-preview-Q4_K_M.gguf --local-dir hf/gguf
python -m export.export_d --gguf hf/gguf/raon-1-mini-preview-Q4_K_M.gguf --hf hf/Raon-1-mini-preview --out weights-d
python -m export.verify_d --manifest weights-d/manifest.json --gguf hf/gguf/raon-1-mini-preview-Q4_K_M.gguf --all
python -m export.verify_d --weights-dir weights-d --expected export/expected_sha256.txt
deactivate
```

The GGUF input is **19,686,434,240 bytes**, SHA256 `ba56ad4bd6d15f8c4255876ff252890143074a5d3835fccfc656cee915999ee8`. Download the complete HF model, not only its safetensors. Export copies `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` and model configuration to the output; missing tokenizer assets cause an error. The expected-checksum comparison reports missing, extra and changed files, excludes timestamped `manifest.json`, and exits nonzero on any mismatch. If converting elsewhere, copy the **entire** `weights-d` directory to the laptop.

### 4. First build and selfcheck

Back in the repository on the Windows laptop:

```powershell
.\scripts\run.cmd scripts\selfcheck.py --weights-d .\weights-d
```

The first selfcheck in a fresh checkout builds 64 xclbin files and took **257 seconds** on the test laptop; a later run took **62 seconds** and built no new files. The resulting build directory contained 7,459 files and used 531 MB. Byte comparisons with bundle sizes 0, 16, 64 and 256, including the direct decode path, did not add build files. Selfcheck audits cached object copies before loading, uses `RAON_PREFILL_CHUNK=256` if unset, and preserves an explicitly supplied value. It runs the fixed English review prompt, greedy generation, max_tokens 32. Expected SHA256 prefix of `thinking + "|" + text`: **`3dc9191d4d3d6344`**. It prints loading/prefill/decode time, generated tokens, and `RESULT: PASS` or `RESULT: FAIL` (exit 0 or 1). `--model-dir` defaults to `--weights-d`.

### 5. Run the server

```powershell
.\serve\run-serve.ps1 -WeightsD .\weights-d
```

For PowerShell execution-policy help, see [step 2](#2-install-the-toolchain).

Defaults: `127.0.0.1:11434`, keep-alive **300 seconds**. The process stays in the foreground and passes its exit code back to PowerShell.

Optional: start automatically at logon (Interactive/Limited, current user):

```powershell
.\scripts\install-task.ps1 -WeightsD .\weights-d -Start
# Remove the task and launcher, and stop the server started by that task:
.\scripts\install-task.ps1 -Uninstall
```

The default task name is `Raon NPU server`; launcher and logs live in `%LOCALAPPDATA%\raon-npu`. `-IronDir`, `-Port`, `-ListenAddress`, `-KeepAlive`, `-LogDir`, `-StateDir` and `-TaskName` are configurable. The task starts on battery, does not stop on battery, has priority 7 and no execution time limit, retries three times at one-minute intervals, and ignores duplicate starts. Registration replaces the named task; use one task per state directory. Paths containing spaces are supported.

## API

See [the detailed Korean server documentation](serve/README.md).

| Method | Endpoint | Purpose |
|---|---|---|
| GET / HEAD | `/` | Ollama liveness |
| GET | `/healthz` | Health |
| GET | `/api/version`, `/api/tags`, `/api/ps` | Ollama metadata and loaded models |
| POST | `/api/show` | Model metadata |
| POST | `/api/chat`, `/api/generate` | Ollama chat / generation |
| GET | `/v1/models` | OpenAI model list |
| POST | `/v1/chat/completions`, `/v1/completions` | OpenAI chat / completion |

```powershell
'{"model":"raon-1-mini","prompt":"Hello","stream":false}' | curl.exe http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" --data-binary "@-"
```

## Update

```powershell
git pull
.\scripts\run.cmd scripts\kernel_object_audit.py --fix
.\scripts\run.cmd scripts\selfcheck.py --weights-d .\weights-d
```

Stop the server before updating or fixing cached objects. On Windows, IRON copies kernel objects into design directories and does not refresh existing copies; `--fix` removes stale copies and their dependency files, then touches the source objects for rebuilding. Without `--fix` the audit only reads. Use `--repo` for another checkout and `--json-out` for a JSON report.

## Security

There is **no authentication or TLS**. By default, only this PC can connect (`127.0.0.1`). To deliberately open a trusted LAN, pass `-ListenAddress 0.0.0.0`; both launch scripts warn about the exposure. Do not expose the server directly to the Internet. Request logs (`serve.log`) may contain prompts.

## Troubleshooting

| Symptom | Action |
|---|---|
| First execution is slow | Wait for kernel compilation; keep the console/log visible. |
| Stale object audit fails | Stop the server, run the update audit with `--fix`, then selfcheck. |
| WinError 10048 | Another process owns the port; stop it or choose `-Port`. |
| Out of memory | Close other memory-heavy programs; tested RAM is 64 GB and loaded usage is about 25 GB. |
| Request after idle is slow | Weights unload after 300 s; the next request loads them again. Adjust `-KeepAlive` if needed. |
| Missing weights / tokenizer | Supply `-WeightsD` / `--weights-d`; copy the whole converted directory and rerun verification. |

## Environment variables

| Variable | Purpose / default |
|---|---|
| `RAON_WEIGHTS_D` | Weights path for server launcher and selfcheck; required if no CLI path |
| `RAON_MODEL_DIR`, `RAON_GOLDEN_DIR` | Host check input paths; otherwise supply CLI arguments |
| `RAON_IRON_DIR` | IRON checkout; launchers fall back to `C:\dev\IRON` |
| `RAON_MLIR_AIE_DIR` | mlir-aie checkout; runner falls back to `C:\dev\mlir-aie`; optional codec tests require it |
| `RAON_PREFILL_CHUNK` | Launch scripts/selfcheck set 256 only when empty; engine code default stays 0 |
| `RAON_GGUF` | Optional real GGUF export-test fixture |
| `RAON_SOURCE_DIR`, `RAON_SRC_DIR` | Optional HF model Python sources for golden/host tests |
| `RAON_IRON_PATH` | Optional reference IRON import path for quantization tests |
| `RAON_MLIR_AIE_PY`, `RAON_IRON_PY` | Optional Python source trees for API compatibility tests |
| `RAON_TXN_PATH` | Optional static GEMV transaction binary for development tests |

Other engine tuning variables keep their existing defaults; this packaging does not change numerical behavior.

## Repository layout

| Directory | Contents |
|---|---|
| `quant/` | CPU quantization, packing and quality evaluation |
| `export/` | D-mode conversion, manifests and byte verification |
| `golden/` | CPU model snapshots and RDA reference checks |
| `kernels/` | AIE kernel sources and device designs |
| `bench/` | Development benchmarks; PowerShell scripts assume a `C:\dev` layout |
| `host/` | NumPy reference model and weights |
| `npu/` | NPU backend; development PowerShell scripts assume a `C:\dev` layout |
| `serve/` | Ollama/OpenAI-compatible HTTP server and manual launcher |
| `scripts/` | Setup, selfcheck, object audit and optional logon task |

Subdirectory documentation remains in Korean where originally written in Korean.

Some code comments refer to section numbers in unpublished internal design notes (`DESIGN-*.md`). They do not affect behavior.

## License and attribution

See [Apache-2.0 LICENSE](LICENSE), [NOTICE](NOTICE), and [LLVM exception](LICENSES/LLVM-exception.txt) for source attribution. KRAFTON Raon-1-mini-preview weights are separately distributed under Apache-2.0, and are not included here. KRAFTON, AMD and other trademarks belong to their respective owners.
