# ALPS: Activation-Based Length Prediction for LLM Scheduling

Prompt dataset for the ALPS paper.

## Dataset

`prompts.txt` contains 180 curated prompts across three response length categories:

| Category | Count | Target Length | Types |
|----------|-------|---------------|-------|
| Short | 60 | 50-150 tokens | Factual, yes/no, definitions, simple questions |
| Medium | 65 | 200-400 tokens | Explanations, comparisons, summaries, how-to |
| Long | 55 | 500-1000 tokens | Detailed explanations, multi-part, analytical, creative |

All prompts are designed to elicit natural-length responses (EOS termination, not max-token truncation).

## Results Summary

| Model | R² (test) | MAE | 
|-------|-----------|-----|
| Gemma-2-9B | 0.943 | 38 tokens |
| Llama-3.1-8B | 0.880 | 71 tokens |
| Qwen-2.5-7B | 0.857 | 80 tokens |

## Code

Experimental code available from the author upon request.

## Citation

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
