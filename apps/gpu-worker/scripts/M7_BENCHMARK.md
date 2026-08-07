# M7.0 benchmark runner

`m7_benchmark.py` creates a sanitized, versioned measurement artifact. It is not a GPU
acceptance gate and never decides whether the 10-minute KPI passed. In particular, a successful
CPU-only run has `measurement_status: "completed"` but `evidence.gpu.status: "unavailable"`,
`claims.gpu_acceptance: false`, and `claims.kpi_evaluated: false`.

Run it from `apps/gpu-worker` on the machine or Pod that owns the GPU:

Replace every `REPLACE_*` token with measured fixture metadata before running; placeholders are
deliberately not accepted as report evidence.

After the model cache is complete, export the exact nine-component runtime manifest. This is a
filesystem-only validation/export and does not require `HF_TOKEN` or contact Hugging Face. The
cache marker also rechecks the post-preload file inventory for every resolved snapshot and the
explicit pyannote dependency graph before it permits this smaller benchmark manifest to be written:

```bash
python scripts/preload_models.py \
  --export-m7-manifest /workspace/benchmarks/model-manifest.json
```

```bash
uv run --frozen python -m scripts.m7_benchmark \
  --scenario analyze-1m-2speaker \
  --output /workspace/benchmarks/analyze-1m-2speaker.json \
  --input-duration-seconds 60 \
  --input-size-bytes 1920044 \
  --input-sha256 REPLACE_WITH_THE_FIXTURE_SHA256 \
  --speaker-count 2 \
  --language Vietnamese \
  --sample-rate-hz 16000 \
  --channels 1 \
  --warmups 1 \
  --iterations 5 \
  --model-manifest /workspace/benchmarks/model-manifest.json \
  --require-gpu \
  -- python -m your_bounded_live_benchmark
```

Exercise the report contract on a CPU workstation without making a GPU claim:

```bash
uv run --frozen python -m scripts.m7_benchmark \
  --scenario cpu-contract \
  --output ./cpu-contract.json \
  --input-duration-seconds 1 \
  --input-size-bytes 32044 \
  --input-sha256 REPLACE_WITH_THE_FIXTURE_SHA256 \
  --speaker-count 1 \
  --language English \
  --warmups 0 \
  --iterations 2 \
  -- python -c "pass"
```

This CPU command exits `0` when both measurements complete, while the report still says GPU
evidence is unavailable. The CLI exits `2` after writing a report when execution fails, an
observation is invalid, or `--require-gpu` cannot verify CUDA plus `nvidia-smi`; it never converts
that condition into a skip or pass. Invalid CLI configuration exits through `argparse` with `2`
before a report can be constructed.

Input/model metadata, the report destination, and the first device-wide VRAM sample are validated
before a workload starts. Each workload is isolated in a POSIX process group or Windows Job Object.
Timeout, VRAM sampler failure, interruption, and a parent that exits while descendants remain all
trigger verified tree cleanup. Failure to prove cleanup has a distinct fail-closed error code.
Every successful measurement iteration must expose the same latency and VRAM stage set; a coverage
mismatch fails the report instead of producing percentiles from a partial sample.

The report records bounded input facts plus the fixture SHA-256 needed for reproducibility, never a
filename, source path, URL, transcript,
translation, job ID, GPU UUID, PID, environment variable, command line, or credential. The CLI
summary likewise prints only status and `output_written: true`.

## Child observation protocol

For every invocation, the runner sets:

- `OMNIVOICE_BENCHMARK_OBSERVATION_PATH`: unique output path in a temporary directory;
- `OMNIVOICE_BENCHMARK_RUN_KIND`: `warmup` or `measurement`;
- `OMNIVOICE_BENCHMARK_RUN_INDEX`: one-based index within that kind.

The child may atomically write at most 64 KiB using this exact shape:

```json
{
  "schema_version": 1,
  "stage_latencies_ms": {
    "download": 81.5,
    "demucs": 1240.0,
    "asr": 2800.25,
    "translation": 930.0
  },
  "gpu_memory": {
    "schema_version": 1,
    "source": "nvidia-smi",
    "scope": "visible_devices_total",
    "device_count": 1,
    "sample_interval_ms": 100,
    "capacity_bytes": 49457143808,
    "baseline_used_bytes": 8589934592,
    "final_used_bytes": 8697308774,
    "peak_used_bytes": 12884901888,
    "stages": [
      {
        "stage": "Demucs",
        "start_used_bytes": 8589934592,
        "peak_used_bytes": 12884901888,
        "end_used_bytes": 9663676416,
        "sample_count": 14
      }
    ]
  }
}
```

`gpu_memory` may be `null`. The runner independently samples aggregate visible-device VRAM around
the complete command. Child stage metrics are accepted only when topology matches and no child peak
exceeds that independent envelope. Any extra field or invalid bound fails the run with a fixed,
sanitized error code.

Use `--model-manifest` when the deployed cache can export exact resolved commits. The input is a
strict array of `{ "component", "model_id", "revision" }`; these entries are labeled
`declared_runtime_snapshot`, not silently promoted to source-verified pins. Without it, the runner
records repository pins for AudioSeal and MOSS and marks Qwen, Whisper, pyannote, and Demucs
revisions `unresolved`. The cache export uses `adefossez/HTDemucs`, the actual Hub repository,
instead of the loader alias `hf://htdemucs`. Paths, URLs, query strings, and free-form revision
values are rejected.

The machine-readable report contract is
[`m7_benchmark_report.schema.json`](m7_benchmark_report.schema.json). Keep raw reports outside Git;
publish only sanitized artifacts after reviewing the hardware/model evidence fields.
