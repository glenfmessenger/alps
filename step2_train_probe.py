"""
Cost Prediction PoC v2 - Train Probes

Trains on both:
1. All data (includes truncated)
2. EOS-only data (clean, naturally terminated)

Reports results for both to compare.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProbeMetrics:
    """Metrics for a trained probe."""
    name: str
    r2_train: float
    r2_test: float
    r2_cv: float
    mae_test: float
    rmse_test: float
    correlation: float
    n_samples: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 42,
) -> ProbeMetrics:
    """Train a single probe and return metrics."""
    
    if len(X) < 20:
        return ProbeMetrics(
            name=name, r2_train=0, r2_test=0, r2_cv=0,
            mae_test=0, rmse_test=0, correlation=0, n_samples=len(X)
        )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha)),
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    cv_scores = cross_val_score(
        Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))]),
        X, y, cv=min(5, len(X) // 5), scoring='r2'
    )
    r2_cv = cv_scores.mean()
    
    correlation = np.corrcoef(y_test, y_pred_test)[0, 1] if len(y_test) > 1 else 0
    
    return ProbeMetrics(
        name=name,
        r2_train=r2_train,
        r2_test=r2_test,
        r2_cv=r2_cv,
        mae_test=mae_test,
        rmse_test=rmse_test,
        correlation=correlation,
        n_samples=len(X),
    )


def train_all_probes(
    activations: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    metadata: Dict,
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 42,
    prefix: str = "",
) -> Dict[str, ProbeMetrics]:
    """Train all probe variants."""
    
    results = {}
    n_samples = activations.shape[0]
    n_layers = activations.shape[1]
    
    # Flatten activations
    all_acts = activations.reshape(n_samples, -1)
    
    # Input length only (baseline)
    input_tokens = features[:, -2].reshape(-1, 1)
    metrics = train_probe(input_tokens, targets, f"{prefix}baseline_input_length", alpha, test_size, seed)
    results[f"{prefix}baseline_input_length"] = metrics
    print(f"  {metrics.name}: R²={metrics.r2_test:.4f} (CV={metrics.r2_cv:.4f})")
    
    # Features only
    metrics = train_probe(features, targets, f"{prefix}features_only", alpha, test_size, seed)
    results[f"{prefix}features_only"] = metrics
    print(f"  {metrics.name}: R²={metrics.r2_test:.4f} (CV={metrics.r2_cv:.4f})")
    
    # Last layer
    last_layer = activations[:, -1, :]
    metrics = train_probe(last_layer, targets, f"{prefix}last_layer", alpha, test_size, seed)
    results[f"{prefix}last_layer"] = metrics
    print(f"  {metrics.name}: R²={metrics.r2_test:.4f} (CV={metrics.r2_cv:.4f})")
    
    # All layers
    metrics = train_probe(all_acts, targets, f"{prefix}all_layers", alpha, test_size, seed)
    results[f"{prefix}all_layers"] = metrics
    print(f"  {metrics.name}: R²={metrics.r2_test:.4f} (CV={metrics.r2_cv:.4f})")
    
    # Per layer
    layer_indices = metadata.get("layer_indices", list(range(n_layers)))
    for i, layer_idx in enumerate(layer_indices):
        layer_acts = activations[:, i, :]
        metrics = train_probe(layer_acts, targets, f"{prefix}layer_{layer_idx}", alpha, test_size, seed)
        results[f"{prefix}layer_{layer_idx}"] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Cost Prediction Probes v2")
    parser.add_argument("--data-dir", type=str, default="./cost_prediction_v2_data")
    parser.add_argument("--output-dir", type=str, default="./cost_prediction_v2_results")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ALL data
    print("Loading ALL data...")
    activations = np.load(data_dir / "activations.npy")
    features = np.load(data_dir / "features.npy")
    targets = np.load(data_dir / "targets.npy")
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(targets)} samples")
    print(f"Target range: {targets.min()} - {targets.max()}, mean={targets.mean():.0f}")
    
    # Train on ALL data
    print("\n" + "=" * 60)
    print("TRAINING ON ALL DATA")
    print("=" * 60)
    all_results = train_all_probes(
        activations, features, targets, metadata,
        args.alpha, args.test_size, args.seed, prefix="all_"
    )
    
    # Load EOS-only data if exists
    eos_results = {}
    eos_file = data_dir / "activations_eos.npy"
    if eos_file.exists():
        print("\n" + "=" * 60)
        print("TRAINING ON EOS-ONLY DATA (naturally terminated)")
        print("=" * 60)
        
        activations_eos = np.load(data_dir / "activations_eos.npy")
        features_eos = np.load(data_dir / "features_eos.npy")
        targets_eos = np.load(data_dir / "targets_eos.npy")
        
        with open(data_dir / "metadata_eos.json") as f:
            metadata_eos = json.load(f)
        
        print(f"Loaded {len(targets_eos)} EOS samples")
        print(f"Target range: {targets_eos.min()} - {targets_eos.max()}, mean={targets_eos.mean():.0f}")
        
        eos_results = train_all_probes(
            activations_eos, features_eos, targets_eos, metadata_eos,
            args.alpha, args.test_size, args.seed, prefix="eos_"
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_best = max(all_results.values(), key=lambda x: x.r2_test)
    all_baseline = all_results.get("all_baseline_input_length")
    
    print(f"\nALL DATA ({len(targets)} samples):")
    print(f"  Baseline R²: {all_baseline.r2_test:.4f}" if all_baseline else "  Baseline: N/A")
    print(f"  Best probe: {all_best.name}")
    print(f"  Best R² (test): {all_best.r2_test:.4f}")
    print(f"  Best R² (CV): {all_best.r2_cv:.4f}")
    print(f"  MAE: {all_best.mae_test:.1f} tokens")
    
    if eos_results:
        eos_best = max(eos_results.values(), key=lambda x: x.r2_test)
        eos_baseline = eos_results.get("eos_baseline_input_length")
        
        print(f"\nEOS-ONLY ({len(targets_eos)} samples):")
        print(f"  Baseline R²: {eos_baseline.r2_test:.4f}" if eos_baseline else "  Baseline: N/A")
        print(f"  Best probe: {eos_best.name}")
        print(f"  Best R² (test): {eos_best.r2_test:.4f}")
        print(f"  Best R² (CV): {eos_best.r2_cv:.4f}")
        print(f"  MAE: {eos_best.mae_test:.1f} tokens")
    
    # Success criteria
    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA")
    print("-" * 60)
    
    # Use EOS results if available, otherwise all
    eval_results = eos_results if eos_results else all_results
    eval_best = max(eval_results.values(), key=lambda x: x.r2_test)
    eval_baseline = eval_results.get("eos_baseline_input_length") or eval_results.get("all_baseline_input_length")
    
    meaningful = eval_best.r2_test > 0.30
    useful = eval_best.r2_test > 0.50
    beats_baseline = eval_best.r2_test > (eval_baseline.r2_test + 0.05) if eval_baseline else True
    
    print(f"R² > 0.30 (meaningful): {'✓ PASSED' if meaningful else '✗ FAILED'} ({eval_best.r2_test:.3f})")
    print(f"R² > 0.50 (useful): {'✓ PASSED' if useful else '✗ FAILED'} ({eval_best.r2_test:.3f})")
    print(f"Beats baseline by >5%: {'✓ PASSED' if beats_baseline else '✗ FAILED'}")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    
    if eval_best.r2_test > 0.70:
        print("STRONG SIGNAL: Activations reliably predict generation length.")
        print("→ Proceed with paper draft and system prototype.")
    elif eval_best.r2_test > 0.50:
        print("GOOD SIGNAL: Activations meaningfully predict generation length.")
        print("→ Publishable result. Consider additional experiments for robustness.")
    elif eval_best.r2_test > 0.30:
        print("MODERATE SIGNAL: Some predictive power detected.")
        print("→ Worth investigating further. May need more data or different features.")
    else:
        print("WEAK SIGNAL: Limited predictive power.")
        print("→ Consider different approach or document negative result.")
    
    # Save results
    combined_results = {**all_results, **eos_results}
    results_dict = {name: metrics.to_dict() for name, metrics in combined_results.items()}
    
    summary = {
        "all_data": {
            "n_samples": len(targets),
            "target_mean": float(targets.mean()),
            "target_std": float(targets.std()),
            "best_probe": all_best.name,
            "best_r2_test": all_best.r2_test,
            "best_r2_cv": all_best.r2_cv,
        },
    }
    
    if eos_results:
        summary["eos_only"] = {
            "n_samples": len(targets_eos),
            "target_mean": float(targets_eos.mean()),
            "target_std": float(targets_eos.std()),
            "best_probe": eos_best.name,
            "best_r2_test": eos_best.r2_test,
            "best_r2_cv": eos_best.r2_cv,
        }
    
    summary["success_criteria"] = {
        "meaningful_threshold": meaningful,
        "useful_threshold": useful,
        "beats_baseline": beats_baseline,
    }
    
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump({
            "probes": results_dict,
            "summary": summary,
            "config": {
                "alpha": args.alpha,
                "test_size": args.test_size,
                "seed": args.seed,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
