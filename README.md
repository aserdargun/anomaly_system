# Task 001: LLM-Driven Anomaly Detection with Isolation Forest

An anomaly detection system running entirely on a **Mac Mini M4**. Qwen 3.5 9B (Ollama + Metal GPU) orchestrates the ML pipeline through tool calling, reads generated visualizations via vision, and produces experiment reports. The ML core uses scikit-learn's Isolation Forest with F1 score as the primary metric.

```
macOS Terminal (Mac Mini M4)
    |
Qwen 3.5 9B (Ollama + Metal GPU) -- Agent Loop
    | tool calls
    v
┌─────────────────────────────────┐
│  Tools (Python ARM64)           │
│  generate_synthetic_data        │
│  load_and_preprocess_data       │
│  train_isolation_forest         │
│  compute_metrics                │
│  generate_visualizations        │
│  analyze_chart (vision)         │
│  run_grid_search                │
│  predict_new_data               │
│  generate_report                │
│  update_log                     │
└─────────────────────────────────┘
    | results
    v
log.md / results/
```

---

## Prerequisites (Mac Mini M4 Setup)

Your Mac Mini M4 runs macOS on Apple Silicon with unified memory and Metal GPU. Everything runs locally — no cloud APIs, no CUDA, no discrete GPU.

### 1. Install Homebrew

If you don't have it yet:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Homebrew installs to `/opt/homebrew/` on Apple Silicon and manages ARM64 native packages.

### 2. Install uv (Python project manager)

```bash
brew install uv
```

Verify with `uv --version`. uv replaces pip, venv, and pyproject management in a single ARM64 native binary. It can also install Python itself.

### 3. Install Ollama (local LLM server)

```bash
brew install ollama
```

Or download the macOS app from https://ollama.com/download/mac. Ollama uses Metal to run LLMs on the M4's 10-core GPU. It auto-starts as a macOS LaunchAgent — you don't need to run `ollama serve` manually.

### 4. Pull Qwen 3.5 9B

```bash
ollama pull qwen3.5:latest
```

This downloads the quantized model (~5-6GB). On your M4 with unified memory, the model loads directly into the shared memory pool used by both CPU and GPU. Verify with:

```bash
ollama list
# Should show qwen3.5:latest

ollama run qwen3.5 "Hello, are you working?"
# Quick smoke test -- should respond in 1-3 seconds (Metal accelerated)
```

### 5. Verify everything is running

```bash
# Check Ollama is serving (auto-started on macOS)
curl http://localhost:11434/api/tags
# Should return JSON with your models

# Check architecture
uname -m
# Should show: arm64
```

If Ollama isn't responding: `brew services restart ollama`

### Memory budget on your M4:

```
Total unified memory:     16GB (or 24GB if you have that config)
macOS + system:           ~3GB
Ollama + Qwen 3.5:       ~6GB
Python ML pipeline:       ~2-3GB
Free headroom:            ~4-8GB (plenty for this task)
```

---

## Project Structure

```
task_001_anomaly_detection/
|
|-- README.md                  <-- You are here
|-- task.md                    <-- SMART task definition, M4 hardware specs, architecture
|-- brainstorming_v1.md        <-- Design rationale, M4 advantages, tool calling patterns
|-- plan_v1.md                 <-- Detailed implementation plan with file-by-file specs
|-- log.md                     <-- Experiment log (updated after each run)
|-- claude_code_prompt.md      <-- Copy-paste prompt for Claude Code to build the system
|
|-- experiments/
|   |-- exp_001.md             <-- Experiment 001 template (Isolation Forest default baseline)
|
|-- code/
|   |-- repo.md                <-- Tech stack, prerequisites, how to run
|   |-- build_notes.md         <-- macOS M4 setup, Ollama API reference, Metal notes
|   |-- anomaly_system/        <-- Implementation (see Code Structure below)
|
|-- results/
|   |-- best_solution.md       <-- Best model config and findings
|   |-- benchmarks.md          <-- Comparison table across all experiments
```

### Code Structure

```
code/anomaly_system/
    pyproject.toml
    fix_venv.sh              # Fix for iCloud path spaces

    src/anomaly_system/
        __init__.py
        __main__.py
        main.py              # CLI entry point
        config.py            # Paths, Ollama settings, IF defaults

        llm/
            ollama_client.py     # Async Ollama HTTP client (Metal)
            tool_registry.py     # Maps Python functions → LLM tool schemas
            agent.py             # Core agent loop
            vision_analyzer.py   # Chart image analysis via LLM vision
            report_generator.py  # LLM-generated experiment reports

        tools/
            data_loader.py       # Load, preprocess, synthetic data generation
            data_utils.py        # Scaling, splitting, rolling features
            isolation_forest.py  # Train and predict
            evaluation.py        # F1, precision, recall, ROC-AUC
            visualization.py     # Confusion matrix, score distribution, PR curve
            grid_search.py       # Exhaustive hyperparameter search
            inference.py         # Predict on new data (also standalone CLI)

        data/                # Datasets
        outputs/
            models/          # Saved joblib models
            plots/           # PNG visualizations
            reports/         # Markdown experiment reports
```

---

## Setup

```bash
cd code/anomaly_system
uv sync
./fix_venv.sh  # required — iCloud paths with spaces break Python's .pth files
```

---

## Usage

### CLI Options

| Flag | Description |
|------|-------------|
| `--health` | Check Ollama, Metal, model, memory |
| `--agent` | LLM agent orchestration (default) |
| `--pipeline` | Direct ML pipeline, no LLM |
| `--vision-test` | Analyze existing plots with LLM vision |
| `--think` | Enable Qwen 3.5 thinking mode (default, slower, more reasoning) |
| `--no-think` | Disable thinking mode (faster responses) |
| `--goal` | Custom goal for the agent |
| `--data-path` | Path to input dataset |

### Commands

```bash
cd code/anomaly_system

# Check system readiness
uv run python -m anomaly_system --health

# Full LLM agent pipeline (default, thinking on)
uv run python -m anomaly_system --agent

# Fast mode — disable Qwen thinking for quicker responses
uv run python -m anomaly_system --agent --no-think

# Direct ML pipeline without LLM (fallback)
uv run python -m anomaly_system --pipeline

# Test vision analysis on existing plots
uv run python -m anomaly_system --vision-test

# Custom goal
uv run python -m anomaly_system --agent --goal "Run grid search to find best hyperparameters"

# Standalone inference on new data
uv run python -m anomaly_system.tools.inference --input data/new_data.csv
```

### Thinking Mode

Qwen 3.5 has a built-in thinking/reasoning mode. When enabled (default), the model reasons internally before responding — more accurate but slower. Disable with `--no-think` for faster agent runs where speed matters more than reasoning depth.

```bash
# Default: thinking on (more reasoning, slower)
uv run python -m anomaly_system --agent

# Thinking off (faster, recommended for routine experiments)
uv run python -m anomaly_system --agent --no-think
```

---

## Implementation Guide

### Phase 1: Build the Code with Claude Code

The `claude_code_prompt.md` file contains a complete prompt that tells Claude Code exactly what to build.

1. Open Terminal and navigate to this task folder
2. Run `claude`
3. Paste the content from `claude_code_prompt.md` (everything between `## PROMPT START` and `## PROMPT END`)
4. After Claude Code finishes, verify: `ls code/anomaly_system/src/anomaly_system/`

### Phase 2: Verify the Environment

5. Install dependencies: `cd code/anomaly_system && uv sync && ./fix_venv.sh`
6. Run health check: `uv run python -m anomaly_system --health`

Should print:

```
[Health] Checking system...
  Platform: Darwin arm64 (Apple Silicon)
  Unified Memory: 24.0 GB
  Ollama: running
  Model: qwen3.5:latest
  Metal: yes

[Health] Mac Mini M4 ready!
```

### Phase 3: Run the System

Start with pipeline mode to verify ML code, then move to agent mode.

7. **Pipeline mode (no LLM, ML only):**

```bash
uv run python -m anomaly_system --pipeline
```

Generates synthetic data, trains Isolation Forest with all M4 cores, computes metrics, generates plots. No Ollama needed.

8. **Agent mode (LLM orchestrates everything):**

```bash
uv run python -m anomaly_system --agent --no-think
```

Qwen 3.5 receives a goal, decomposes it into tool calls, runs the ML pipeline, reads charts with vision, and writes a report. The full pipeline completes in 2-5 minutes on M4.

9. **Vision test (chart reading only):**

```bash
uv run python -m anomaly_system --vision-test
```

### Phase 4: Run Experiments

10. **Experiment 1 — Default baseline** (the first agent run from Step 8)

11. **Experiment 2 — Grid search:**

```bash
uv run python -m anomaly_system --agent --no-think --goal "Run a grid search over Isolation Forest hyperparameters: n_estimators [50, 100, 200, 500], contamination [0.01, 0.05, 0.1, auto], max_samples [auto, 256, 512]. Train and evaluate each combination using F1 score. Report the best parameters and their metrics. Generate visualizations for the best model and analyze them."
```

12. **Experiment 3 — Feature engineering:**

```bash
uv run python -m anomaly_system --agent --no-think --goal "Run an anomaly detection experiment with Isolation Forest using the best parameters from the previous grid search. This time, enable feature engineering with window sizes [10, 30, 60] to add rolling statistics. Compare the results against the baseline. Generate visualizations and a comprehensive report."
```

### Phase 5: Custom Dataset

```bash
cp ~/path/to/your/sensor_data.csv code/anomaly_system/src/anomaly_system/data/
uv run python -m anomaly_system --agent --no-think --data-path src/anomaly_system/data/sensor_data.csv
```

Your CSV should have numeric sensor columns and a `label` column (0 = normal, 1 = anomaly).

### Phase 6: Iterate (Recursive Improvement Loop)

After Experiments 1-3:

1. Feed `brainstorming_v1.md`, `plan_v1.md`, and `log.md` back into brainstorming
2. Generate `brainstorming_v2.md` → `plan_v2.md`
3. Implement plan_v2 with Claude Code (add LSTM Autoencoder or TranAD tools)

```
brainstorming_v1 -> plan_v1 -> implement -> log -> brainstorming_v2 -> plan_v2 -> ...
```

---

## Baseline Results (Synthetic Data)

| Metric | Value |
|--------|-------|
| F1 | 0.8696 |
| Precision | 0.7692 |
| Recall | 1.0000 |
| ROC-AUC | 1.0000 |

Model: `IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1)`
Training time: 0.05s on M4

---

## Key File Reference

| When you need to... | Read this file |
|---------------------|---------------|
| Understand the task goal and M4 architecture | `task.md` |
| Understand design decisions and M4 rationale | `brainstorming_v1.md` |
| See the full implementation plan | `plan_v1.md` |
| Build the code with Claude Code | `claude_code_prompt.md` |
| Set up the macOS environment and run commands | `code/repo.md` |
| Understand M4 specifics, Ollama/Metal, and APIs | `code/build_notes.md` |
| Track experiment results | `log.md` |
| See the best model configuration | `results/best_solution.md` |
| Compare all experiments | `results/benchmarks.md` |

---

## Troubleshooting

**iCloud path spaces:** Python's `.pth` editable install breaks on paths with spaces. Run `./fix_venv.sh` after every `uv sync` to fix. This installs a `sitecustomize.py` that adds the `src/` path correctly.

**Ollama not responding:** Ollama auto-starts on macOS. Restart with `brew services restart ollama`.

**Model not found:** Run `ollama pull qwen3.5:latest`. Check with `ollama list`.

**uv sync fails:** Run `uv python install 3.12` to install ARM64 Python.

**Vision analysis timeout:** Complex charts may exceed the 180s timeout. Use `--no-think` to reduce latency.

**Tool calling errors:** Qwen 3.5 may occasionally produce malformed JSON. The agent retries up to 3 times. Check tool schemas in `tools/__init__.py` if it persists.

**Vision analysis seems wrong:** Cross-check against actual numeric metrics. Increase `PLOT_DPI` in `config.py` for clearer charts.

**Memory pressure:** Run `memory_pressure` in Terminal. Close memory-heavy apps if needed. On 24GB M4, not an issue.

**Slow inference:** Check `ollama ps` to verify the model is loaded on GPU. Should be ~15-30 tok/s on M4 Metal.

**matplotlib display issues:** Code uses `Agg` backend for headless rendering. Change `MATPLOTLIB_BACKEND` to `"macosx"` in `config.py` for interactive plots.

**Rosetta warnings:** Reinstall with `uv sync --reinstall` to force ARM64.

---

## Tech Stack

| Component | Technology | M4 Notes |
|-----------|-----------|----------|
| Platform | Mac Mini M4, macOS Sequoia | 10-core CPU, 10-core GPU, 16-core Neural Engine |
| Memory | Unified memory (16/24GB) | Shared between CPU, GPU, and Metal — no separate VRAM |
| GPU acceleration | Metal | Ollama uses Metal for LLM inference (not CUDA) |
| Python project manager | uv | ARM64 native binary, installs ARM64 Python |
| Local LLM server | Ollama | macOS native app, Metal-accelerated, auto-starts |
| LLM model | Qwen 3.5 9B | ~6GB quantized, ~15-30 tok/s on M4 Metal |
| ML algorithm | Isolation Forest (scikit-learn) | CPU-only, n_jobs=-1 uses all M4 cores |
| Linear algebra | Apple Accelerate | numpy/scipy use Accelerate (ARM64 BLAS/LAPACK) |
| HTTP client | httpx | Async communication with Ollama API |
| Data processing | pandas, numpy | ARM64 native, Accelerate-optimized |
| Visualization | matplotlib, seaborn | Agg backend for headless macOS rendering |
| Model persistence | joblib | Save/load trained models and scalers |
| Primary metric | F1 Score | Balances precision and recall for imbalanced data |
