# Dataset Instructions

This guide covers everything needed to contribute a new example:

- Naming conventions and folder structure
- Reference file requirements (JSON output format, class-based C++ style)
- How to write the `empty_*` template
- `build_and_run.py` required function signatures and CLI arguments
- Comparison summary JSON format

## Quick Contribution Checklist

1. Create `datasets/example<NNN>_descriptive_name/<platform>/` (NNN is 3 digits, e.g. `example006`)
2. Add `ref_*.cu` or `ref_*.cpp` — full working implementation
3. Add `empty_*.*` — same file with core logic replaced by `// TODO`
4. Add `build_and_run.py` — must implement `build()`, `run()`, `build_and_run()`, `compare()`
5. Register the example's primary metric in `run_eval/perf_verdict.py` (`PERF_METRICS`)
6. Test locally: `python build_and_run.py --source ref_*.cu`
7. Run evaluation: `python scripts/generate_eval_one.py example<NNN>_descriptive_name`

Details for each step follow below.

### Standard Example

[llm-for-gpu-comm/datasets/example001_gpu_comm_single_process/nv](./example001_gpu_comm_single_process/nv)

---

## Naming and Folder Structure

### Naming requirement: zero-padded 3-digit example IDs

Example folders **must** use a **3-digit zero-padded numeric ID**:

```
example<NNN>_<descriptive_name>/      # ✅ correct
```

- `example006_rdma_read_write_rc` ✅
- `example6_rdma_read_write_rc` ❌ (not zero-padded)
- `example06_rdma_read_write_rc` ❌ (only 2 digits)

This keeps lexicographic sort order aligned with numeric order (so `example010` comes after `example009`, not after `example1`) and ensures consistent naming across the dataset directory and evaluation scripts.

### Overall Structure

Each example must contain exactly **three files**:

* `ref_*.*`
  Full implementation of a small GPU communication task
  (e.g., IPC GPU communication, P2P copy, FIFO, RDMA, etc.)

* `empty_*.*`
  A copy of `ref_*.*` with key logic removed and replaced by `// TODO`, intended for AI completion.

* `build_and_run.py`
  A unified script that compiles, runs, and compares:

  * `ref_*.*`
  * `generated_*.*`
* `generated_*.*`

  After running:

  ```bash
  python scripts/generate_eval_one.py <dataset_name>
  ```

  A file named `generated_*.*` will be created inside the corresponding subfolder of the example directory.


### Platform Organization

  Each example must have a clear and descriptive folder name.
  Different variants may exist to support different hardware configurations.

  We support:

  * **GPU platforms**

    * AMD → `amd`
    * NVIDIA → `nv`

  * **RDMA NIC types**

    * InfiniBand → `ib`
    * EFA → `efa`

#### Case 1 — Single GPU Platform

  If the implementation supports only one GPU platform:

  ```
  ├── amd
  │   ├── build_and_run.py
  │   ├── empty_gpu_p2p_comm.cpp
  │   └── ref_gpu_p2p_comm.cpp
  └── nv
      ├── build_and_run.py
      ├── empty_gpu_p2p_comm.cpp
      └── ref_gpu_p2p_comm.cpp
  ```
#### Case 2 — Multiple GPU Platforms

  If the implementation supports multiple GPU platforms:

  ```
  ├── amd_nv
  │   ├── build_and_run.py
  │   ├── empty_fifo_test_unified.cu
  │   └── ref_fifo_test_unified.cu
  ```

  ---

## Reference File Requirements
### Mandatory Requirements of Reference file 
* All required functions must be implemented in exactly **one** reference file:

```
ref_*.cu
ref_*.cpp
ref_*.hip
ref_*.py
```

* All required logic must be implemented as **class methods**
* Include:

  * correctness test
  * performance benchmark
* Test cases must be meaningful (not trivial)
* The program must print **exactly one JSON object** to stdout
* Provide a brief description at the top of the file explaining the function and purpose of the file.

* Use a clean and well-structured code design.
  * For ref_*.cu/ref_*.cpp/ref_*.hip, Use **C++ style**
  * Implement a dedicated function `runTest(...)` to handle testing and performance measurement.
  * Design clean, modular classes that provide core functionality without embedding test or benchmarking logic.
  * Keep implementation logic separate from evaluation logic.

  * You may follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for reference (this is only a recommendation).

### Mandatory JSON Output Format of Reference File 

#### If Performance Measurement Is Required

The JSON must include:

* Units
* A `metrics` list
* Multiple data sizes (when applicable)

Example:

```json
{
  "Correctness": "PASS",
  "data_size_unit": "MB",
  "throughput_unit": "Gbps",
  "latency_unit": "us",
  "metrics": [
    {"data_size": 256, "throughput_avg": 11, "latency_avg": 22},
    {"data_size": 512, "throughput_avg": 11, "latency_avg": 33}
  ]
}
```
#### If Only Correctness Is Required

```json
{
  "Correctness": "PASS"
}
```
#### Optional Metrics

Additional metrics (e.g., MFU) are allowed if units are specified:

```json
{
  "Correctness": "PASS",
  "data_size_unit": "MB",
  "throughput_unit": "Gbps",
  "latency_unit": "us",
  "mfu_unit": "percent",
  "metrics": [
    {"data_size": 256, "throughput_avg": 11, "latency_avg": "", "mfu": 40},
    {"data_size": 512, "throughput_avg": 11, "latency_avg": "", "mfu": 40}
  ]
}
```

If the benchmark does not involve `data_size`, `throughput`, or `latency`, replace them with appropriate metrics.

---

## Compile and Run Script Requirements (`build_and_run.py`)

Each example must include exactly one script:

```
build_and_run.py
```

It must support:

#### 1️⃣ Single File Mode

* Compile and run a single source file
* Optionally generate plots

Required functions:

```python
def build(...)
def run(...)
def build_and_run(...)
```

Example:

```bash
cd llm-for-gpu-comm/datasets/example001_gpu_comm_single_process/nv
python build_and_run.py --source ref_gpu_p2p_comm.cpp
```

---

#### 2️⃣ Compare Mode

Implemented via:

```python
def compare(...)
```

### Mandatory Requirements of build_and_run.py

* Do NOT print raw stdout of reference or generated programs (by default)
* Save all outputs under `--results-dir`
#### Required Saved Outputs

* Two basic plots:

  * latency comparison
  * throughput comparison
* Raw metrics saved as CSV
* Comparison Summary JSON

Example output structure:
```
└── results
    ├── generated_gpu_p2p_comm_..._metrics.csv
    ├── latency_comparison.png
    ├── ref_gpu_p2p_comm_metrics.csv
    ├── summary.json
    ├── throughput_comparison.png
```
Format of Comparison Summary JSON:

`metrics_comparison.{ref,generated}` must be a flat `{metric_name: number}` dict — that is the only structure the unified verdict reads. Verdict fields (`performance`, `performance_detail`, `verdict_scheme`) are written by `run_eval/perf_verdict.py` automatically at exit; do not hand-roll them.

##### If Performance Measurement Is Required (e.g., various types of communication benchmarks)

  ```json
  {
    "generated_source": "...",
    "ref_source": "...",
    "model": "...",
    "pass_iteration": 1,
    "improvement_iteration": 1,
    "data_size_unit": "MB",
    "latency_unit": "us",
    "throughput_unit": "Mbps",
    "metrics_comparison": {
      "ref":       { "throughput_avg": 12.3, "latency_avg": 45.6, "...": "..." },
      "generated": { "throughput_avg": 12.1, "latency_avg": 46.0, "...": "..." }
    },
    "performance": "on_par",
    "performance_detail": {
      "primary_metric": "throughput_avg",
      "direction": "higher",
      "ref": 12.3,
      "generated": 12.1,
      "improvement_pct": -1.63
    },
    "verdict_scheme": "unified_perf_verdict"
  }
  ```

`performance` is one of:
- **comparison verdicts** — `better` (≥+20%), `on_par` (−5..+20%), `degraded` (−40..−5%), `severely_degraded` (<−40%)
- **non-comparison verdicts** — `info_only` (registry says no perf metric), `no_gen_metrics` (gen lacks the primary metric — usually compile/run failed), `no_ref_metrics` (registry's primary key is wrong), `unknown` (example not registered)

`performance_legacy` is added automatically only when the script's old per-example verdict differed from the unified one.

##### If Performance Measurement Is Not Required (e.g., listing device information)

  ```json
  {
    "generated_source": "...",
    "ref_source": "...",
    "model": "...",
    "pass_iteration": 1,
    "metrics_comparison": {
      "ref":       { "compile_success": true, "run_success": true },
      "generated": { "compile_success": true, "run_success": true }
    },
    "performance": "info_only",
    "verdict_scheme": "unified_perf_verdict"
  }
  ```

##### Per-example metric registry (`run_eval/perf_verdict.py`)

Every example **must** have one entry in `PERF_METRICS`, keyed by the `exampleNNN_xxx` directory name. Two forms:

```python
"example003_fifo_device2host": {"primary": "throughput_MBps", "direction": "higher"},
"example004_rdma_nic_info":    "info_only",
```

- `primary` must be a key that **actually appears** in `metrics_comparison.ref` and `metrics_comparison.generated` with a numeric value (run the example once and inspect `summary.json` to confirm — names like `throughput`, `throughput_avg`, `throughput_MBps`, `bandwidth_gbps` are not interchangeable).
- `direction` is `"higher"` (throughput-style) or `"lower"` (latency-style).
- `"info_only"` means the example has no comparable numeric metric; verdict is fixed to `info_only` and never triggers a perf retry.
- Cheating guard: when `direction="lower"`, `ref>0`, and `gen==0`, improvement is pinned to −100% (treated as did-not-measure) instead of a misleading +100%.

#### Required Printed Comparison Summary

The build_and_run.py script must print:

```
PERFORMANCE COMPARISON (...)

[+] data_size_avg: ...
[+] throughput: ...
[+] latency_avg: ...
Performance: same

============================================================
ref       compile_success: True
ref       run_success:     True
generated compile_success: True
generated run_success:     True
performance:               same
============================================================
```

#### build_and_run.py Usage Example

```bash
python build_and_run.py \
  --compare ref_gpu_p2p_comm.cpp \
  generated_gpu_p2p_comm_xxx.cpp
```

#### build_and_run.py Required Function Signatures

```python
def run(executable, verbose=True) -> RunResult: ...
def build(source_file, output_file, compiler, platform, debug=False, arch=None, verbose=True) -> BuildResult: ...
def build_and_run(...) -> BuildAndRunResult: ...
def compare(...) -> Dict[str, Any]: ...
```

#### Required CLI Arguments
Required Command-Line Arguments, The script must support the following arguments:
* `--source`
  Specifies the source file to compile in single-file mode.

* `--output`
  Specifies the name of the generated executable.

* `--arch`
  Specifies the target GPU architecture (e.g., `sm_80` or `gfx90a`).

* `--build-only`
  Compiles the source file without executing it.

* `--run-only`
  Executes an existing executable without compiling.

* `--compiler`
  Manually specifies the compiler path instead of auto-detection.

* `--platform`
  Forces the compilation platform (`hip` or `cuda`).

* `--plot`
  Enables performance plotting after successful execution.

* `--results-dir`
  Specifies the directory for saving CSV files, plots, and summary JSON.

* `--compare`
  Enables comparison mode by building and running a reference and a generated source file.

* You may add additional optional arguments if needed. However, the required arguments listed above must be fully implemented as specified, and **must not be removed or renamed**. You can refer to `llm-for-gpu-comm/datasets/example001_gpu_comm_single_process/nv/build_and_run.py` as an example implementation.

---
# Empty File Requirements

The empty file is generated by removing implementation from `ref_*.*`.

Rules:

* Keep all class definitions and function signatures unchanged
* Remove only the core logic
* Replace implementation with:

```cpp
// TODO
```

* Test code must remain intact
* Optional: add short hints for AI

---

## Additional Principles

* Do not modify files outside the example folder
* Ensure benchmark results are realistic

  * Example: bandwidth should increase with data size
* Update `.gitignore` to prevent committing:

  * binaries
  * object files
  * generated executables
  * result folders
