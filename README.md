# ALPS: Activation-Based Length Prediction for Intelligent LLM Inference Scheduling

**Research Question:** Do prefill activations encode information about how long the model will generate?

**Answer:** Yes. Linear probes on middle-layer activations predict output length with R² = 0.86–0.94 across Llama, Gemma, and Qwen model families.

---

## Dataset Information

The prompt dataset (`prompts_curated.py`) contains **180 manually curated prompts** assembled for this study. Prompts were written by the author to span three response length categories:

| Category | Count | Target Response Length | Task Types |
|----------|-------|------------------------|------------|
| Short | 60 | 50–150 tokens | Factual Q&A, yes/no, definitions, simple lookups |
| Medium | 65 | 200–400 tokens | Explanations, comparisons, summaries, how-to |
| Long | 55 | 500–1000 tokens | Detailed analysis, multi-part, creative writing |

All prompts were designed to elicit naturally terminating responses (EOS-terminated, not max-token-truncated). Prompts were sourced entirely by the author — no external datasets were used. The dataset is provided in `prompts_curated.py` as a Python list, and in `prompts.txt` as a plain-text file with category tags.

Experimental outputs (recorded output token counts per model) are provided in `alps_llama/results/probe_results.json`, `alps_gemma/results/probe_results.json`, and `alps_qwen/results/probe_results.json`.

---

## Code Information

This repository contains the full experimental pipeline for the ALPS paper:

| File | Description |
|------|-------------|
| `run_poc.py` | End-to-end runner (data collection → probe training → analysis) |
| `step1_collect_data.py` | Runs prompts through model, extracts last-token activations at multiple layers, records output lengths |
| `step2_train_probe.py` | Trains ridge regression probes on (activation, output_length) pairs |
| `step3_analyze.py` | Generates result tables and visualisations |
| `prompts_curated.py` | The 180-prompt dataset |

The code was developed and tested with Python 3.10+ on CUDA-enabled hardware.

---

## Usage Instructions

### Installation

```bash
git clone https://github.com/glenfmessenger/alps
cd alps
pip install torch transformers datasets scikit-learn matplotlib numpy tqdm
```

A CUDA-capable GPU with ≥24 GB VRAM is recommended for running full experiments (Gemma-2-9B requires ~18 GB). CPU execution is possible but slow (~10× slower).

### Running Experiments

```bash
# Full pipeline (one command)
python run_poc.py --model meta-llama/Llama-3.1-8B-Instruct --n-prompts 200

# Or step-by-step
python step1_collect_data.py --n-prompts 200 --output-dir ./data
python step2_train_probe.py --data-dir ./data --output-dir ./results
python step3_analyze.py --data-dir ./data --results-dir ./results --output-dir ./analysis
```

### Loading Pre-Computed Results

```python
import json

with open("alps_llama/results/probe_results.json") as f:
    results = json.load(f)

# results contains: r2_test, r2_cv, mae, layer_scores, baseline_r2
print(results["best_layer"]["r2_test"])  # 0.880
```

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Hugging Face `transformers` ≥ 4.40
- `scikit-learn` ≥ 1.3
- `numpy`, `matplotlib`, `tqdm`, `datasets`
- CUDA-capable GPU (≥24 GB VRAM recommended for 9B models)
- Hugging Face account with access to gated models (Llama 3.1 requires approval at meta-llama/Llama-3.1-8B-Instruct)

---

## Key Results

| Model | Best Layer | R² (test) | R² (CV) | MAE | Baseline R² |
|-------|-----------|------------|---------|-----|-------------|
| Llama 3.1 8B | Layer 16 (50%) | 0.880 | 0.884 | 71 tokens | 0.262 |
| Gemma 2 9B | All layers | 0.943 | 0.950 | 38 tokens | 0.396 |
| Qwen 2.5 7B | Layer 14 (50%) | 0.857 | 0.848 | 80 tokens | 0.356 |

---

## Methodology

### Data Collection
- 180 prompts processed through each model using the official chat template
- Last-token hidden states extracted at layers corresponding to 25%, 50%, 75%, and 100% of model depth using PyTorch forward hooks
- Output token count recorded at natural EOS termination (no max-token truncation)

### Probe Training
- Ridge regression (α = 1.0) trained on (activation, output_length) pairs
- StandardScaler normalisation applied to activations
- 80/20 train/test split; 5-fold cross-validation for generalisation estimate
- Baseline: input length only (no activations)

### Evaluation
- R² (coefficient of determination) on held-out test set
- Mean Absolute Error (MAE) in tokens
- Comparison against input-length-only baseline

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{messenger2026alps,
  title={{ALPS}: Activation-Based Length Prediction for Intelligent {LLM} Inference Scheduling},
  author={Messenger, Glen},
  journal={AI},
  publisher={MDPI},
  year={2026}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contribution Guidelines

This repository is the canonical code release for the ALPS paper. Bug reports and reproducibility issues are welcome via GitHub Issues. Feature additions or forks for derivative research are welcome under the terms of the Apache 2.0 licence.
