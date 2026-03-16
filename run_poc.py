#!/usr/bin/env python3
"""
Cost Prediction PoC v2 - Clean Experiment Runner

Uses curated prompts to get varied-length responses without truncation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_step(name: str, command: list) -> int:
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    print(f"Command: {' '.join(command)}\n")
    
    start = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s (exit code {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Cost Prediction PoC v2")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts (balanced across short/medium/long)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max generation (high to avoid truncation)")
    parser.add_argument("--output-dir", type=str, default="./cost_prediction_v2_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-collect", action="store_true")
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    
    print("=" * 70)
    print("COST PREDICTION POC v2 - CLEAN EXPERIMENT")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Prompts: {args.n_prompts} (balanced short/medium/long)")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output: {base_dir}")
    
    start = time.time()
    
    # Data collection
    if not args.skip_collect:
        run_step("Data Collection", [
            sys.executable, "step1_collect_data.py",
            "--model", args.model,
            "--n-prompts", str(args.n_prompts),
            "--max-tokens", str(args.max_tokens),
            "--output-dir", str(data_dir),
            "--seed", str(args.seed),
            "--device", args.device,
        ])
    
    # Train probes
    run_step("Probe Training", [
        sys.executable, "step2_train_probe.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(results_dir),
        "--seed", str(args.seed),
    ])
    
    total = time.time() - start
    print("\n" + "=" * 70)
    print(f"COMPLETE in {total/60:.1f} minutes")
    print("=" * 70)
    print(f"Results: {results_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
