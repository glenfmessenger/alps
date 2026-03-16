# ALPS: Activation-Based Length Prediction for LLM Scheduling

**Research Question:** Do prefill activations encode information about how long the model will generate?

**Answer:** Yes. Linear probes on middle-layer activations predict output length with R² = 0.86-0.94 across Llama, Gemma, and Qwen model families.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets scikit-learn matplotlib numpy tqdm

# Run experiment (180 prompts, ~2-3 hours per model)
python run_poc.py --model meta-llama/Llama-3.1-8B-Instruct --n-prompts 200

# Or run steps individually
python step1_collect_data.py --n-prompts 200 --output-dir ./data
python step2_train_probe.py --data-dir ./data --output-dir ./results
python step3_analyze.py --data-dir ./data --results-dir ./results --output-dir ./analysis
```

## Project Structure

```
├── README.md                 # This file
├── RESULTS_SUMMARY.md        # Detailed results in text format
├── prompts_curated.py        # 180 curated prompts (short/medium/long)
├── run_poc.py                # One-shot runner
├── step1_collect_data.py     # Collect activations + output lengths
├── step2_train_probe.py      # Train linear probes
├── step3_analyze.py          # Generate visualizations
│
├── alps_llama/  # Llama 3.1 8B results
│   └── results/probe_results.json
│
├── alps_gemma/          # Gemma 2 9B results
│   └── results/probe_results.json
│
└── alps_qwen/           # Qwen 2.5 7B results
    └── results/probe_results.json
```

## Key Results

| Model | Best R² (test) | R² (CV) | MAE | Baseline R² |
|-------|----------------|---------|-----|-------------|
| Llama 3.1 8B | 0.880 | 0.884 | 71 tokens | 0.262 |
| Gemma 2 9B | 0.943 | 0.950 | 38 tokens | 0.396 |
| Qwen 2.5 7B | 0.857 | 0.848 | 80 tokens | 0.356 |

## Methodology

1. **Data Collection**: Run 180 diverse prompts through the model
   - Extract last-token activation at layers 25%, 50%, 75%, 100%
   - Record actual output token count (natural EOS termination)

2. **Probe Training**: Ridge regression from activations → token count
   - StandardScaler normalization
   - 80/20 train/test split
   - 5-fold cross-validation

3. **Evaluation**: Compare against baseline (input length only)

## Citation

If you use this code, please cite:

```bibtex
@article{messenger2026alps,
  title={ALPS: Activation-Based Length Prediction for Intelligent LLM Inference Scheduling},
  author={Messenger, Glen},
  year={2026},
  note={Preprint}
}
```

## License

Apache 2.0
