"""
Cost Prediction PoC - Step 3: Analysis and Visualization

Generates detailed analysis and visualizations of the cost prediction results.

Outputs:
- Scatter plots: predicted vs actual
- Error analysis by prompt type
- Layer-wise comparison
- Feature importance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load collected data."""
    activations = np.load(data_dir / "activations.npy")
    features = np.load(data_dir / "features.npy")
    targets = np.load(data_dir / "targets.npy")
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    return activations, features, targets, metadata


def plot_predicted_vs_actual(
    activations: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    alpha: float = 1.0,
    seed: int = 42,
):
    """
    Create scatter plots of predicted vs actual output length.
    """
    # Flatten activations for all-layers probe
    n_samples = activations.shape[0]
    all_acts = activations.reshape(n_samples, -1)
    
    # Train probes
    X_train, X_test, y_train, y_test = train_test_split(
        all_acts, targets, test_size=0.2, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Also train baseline
    input_tokens = features[:, -2].reshape(-1, 1)
    X_train_bl, X_test_bl, y_train_bl, y_test_bl = train_test_split(
        input_tokens, targets, test_size=0.2, random_state=seed
    )
    baseline = Ridge(alpha=alpha)
    baseline.fit(X_train_bl, y_train_bl)
    y_pred_bl = baseline.predict(X_test_bl)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Activation probe predictions
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=20, c='blue')
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Output Tokens')
    ax.set_ylabel('Predicted Output Tokens')
    ax.set_title(f'Activation Probe\nR² = {1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2):.3f}')
    ax.legend()
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    
    # 2. Baseline predictions
    ax = axes[1]
    ax.scatter(y_test_bl, y_pred_bl, alpha=0.3, s=20, c='orange')
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Output Tokens')
    ax.set_ylabel('Predicted Output Tokens')
    ax.set_title(f'Baseline (Input Length)\nR² = {1 - np.sum((y_test_bl - y_pred_bl)**2) / np.sum((y_test_bl - y_test_bl.mean())**2):.3f}')
    ax.legend()
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    
    # 3. Residual distribution
    ax = axes[2]
    residuals_act = y_pred - y_test
    residuals_bl = y_pred_bl - y_test_bl
    
    ax.hist(residuals_bl, bins=50, alpha=0.5, label=f'Baseline (std={residuals_bl.std():.1f})', color='orange')
    ax.hist(residuals_act, bins=50, alpha=0.5, label=f'Activation (std={residuals_act.std():.1f})', color='blue')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Predicted - Actual)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distributions')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'predicted_vs_actual.png'}")


def plot_layer_comparison(
    activations: np.ndarray,
    targets: np.ndarray,
    metadata: Dict,
    output_dir: Path,
    alpha: float = 1.0,
    seed: int = 42,
):
    """
    Compare predictive power across different layers.
    """
    layer_indices = metadata.get("layer_indices", list(range(activations.shape[1])))
    n_layers = len(layer_indices)
    
    r2_scores = []
    
    for i in range(n_layers):
        layer_acts = activations[:, i, :]
        X_train, X_test, y_train, y_test = train_test_split(
            layer_acts, targets, test_size=0.2, random_state=seed
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        r2_scores.append(r2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(n_layers), r2_scores, color='steelblue', edgecolor='black')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'Layer {idx}' for idx in layer_indices], rotation=45, ha='right')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² Score')
    ax.set_title('Predictive Power by Layer')
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add horizontal line for best score
    best_idx = np.argmax(r2_scores)
    ax.axhline(r2_scores[best_idx], color='red', linestyle='--', alpha=0.5,
               label=f'Best: Layer {layer_indices[best_idx]} (R²={r2_scores[best_idx]:.3f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'layer_comparison.png'}")
    
    return dict(zip(layer_indices, r2_scores))


def plot_output_distribution(
    targets: np.ndarray,
    metadata: Dict,
    output_dir: Path,
):
    """
    Visualize the distribution of output lengths.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Histogram
    ax = axes[0]
    ax.hist(targets, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(targets.mean(), color='red', linestyle='--', label=f'Mean: {targets.mean():.1f}')
    ax.axvline(np.median(targets), color='orange', linestyle='--', label=f'Median: {np.median(targets):.1f}')
    ax.set_xlabel('Output Tokens')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Output Lengths')
    ax.legend()
    
    # 2. By stop reason
    ax = axes[1]
    stop_reasons = metadata.get("stop_reasons", [])
    if stop_reasons:
        unique_reasons = list(set(stop_reasons))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_reasons)))
        
        for reason, color in zip(unique_reasons, colors):
            mask = [r == reason for r in stop_reasons]
            subset = targets[mask]
            ax.hist(subset, bins=30, alpha=0.6, label=f'{reason} (n={len(subset)})',
                   color=color, edgecolor='black')
        
        ax.set_xlabel('Output Tokens')
        ax.set_ylabel('Count')
        ax.set_title('Output Length by Stop Reason')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'output_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'output_distribution.png'}")


def plot_input_vs_output(
    features: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
):
    """
    Scatter plot of input length vs output length.
    """
    input_tokens = features[:, -2]  # prompt_tokens column
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(input_tokens, targets, alpha=0.3, s=20, c='steelblue')
    
    # Add trend line
    z = np.polyfit(input_tokens, targets, 1)
    p = np.poly1d(z)
    x_line = np.linspace(input_tokens.min(), input_tokens.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.1f}')
    
    # Correlation
    corr = np.corrcoef(input_tokens, targets)[0, 1]
    ax.set_xlabel('Input Tokens')
    ax.set_ylabel('Output Tokens')
    ax.set_title(f'Input vs Output Length (correlation: {corr:.3f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'input_vs_output.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'input_vs_output.png'}")


def plot_error_analysis(
    activations: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    metadata: Dict,
    output_dir: Path,
    alpha: float = 1.0,
    seed: int = 42,
):
    """
    Analyze prediction errors.
    """
    # Train model
    n_samples = activations.shape[0]
    all_acts = activations.reshape(n_samples, -1)
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        all_acts, targets, np.arange(len(targets)),
        test_size=0.2, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    errors = np.abs(y_pred - y_test)
    relative_errors = errors / (y_test + 1)  # +1 to avoid division by zero
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error vs actual output length
    ax = axes[0, 0]
    ax.scatter(y_test, errors, alpha=0.3, s=20)
    ax.set_xlabel('Actual Output Tokens')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Output Length')
    
    # 2. Relative error vs actual output length
    ax = axes[0, 1]
    ax.scatter(y_test, relative_errors * 100, alpha=0.3, s=20)
    ax.axhline(50, color='r', linestyle='--', alpha=0.5, label='50% error')
    ax.set_xlabel('Actual Output Tokens')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Error vs Output Length')
    ax.set_ylim(0, min(200, relative_errors.max() * 100 * 1.1))
    ax.legend()
    
    # 3. Error by output length bucket
    ax = axes[1, 0]
    buckets = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
    bucket_labels = ['0-50', '50-100', '100-200', '200-300', '300-500']
    bucket_errors = []
    bucket_counts = []
    
    for low, high in buckets:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            bucket_errors.append(errors[mask].mean())
            bucket_counts.append(mask.sum())
        else:
            bucket_errors.append(0)
            bucket_counts.append(0)
    
    bars = ax.bar(bucket_labels, bucket_errors, color='steelblue', edgecolor='black')
    ax.set_xlabel('Output Length Bucket')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Output Length Bucket')
    
    # Add count labels
    for bar, count in zip(bars, bucket_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 4. Worst predictions
    ax = axes[1, 1]
    worst_idx = np.argsort(errors)[-10:]  # 10 worst predictions
    
    ax.barh(range(10), errors[worst_idx], color='coral', edgecolor='black')
    ax.set_yticks(range(10))
    
    # Get prompts for worst predictions
    prompts = metadata.get("prompts", [])
    worst_labels = []
    for i in worst_idx:
        actual_idx = idx_test[i]
        if prompts and actual_idx < len(prompts):
            prompt_preview = prompts[actual_idx][:40] + "..."
        else:
            prompt_preview = f"Sample {actual_idx}"
        worst_labels.append(f"{prompt_preview} (pred={y_pred[i]:.0f}, actual={y_test[i]:.0f})")
    
    ax.set_yticklabels(worst_labels, fontsize=8)
    ax.set_xlabel('Absolute Error')
    ax.set_title('10 Worst Predictions')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'error_analysis.png'}")


def generate_report(
    results_file: Path,
    output_dir: Path,
    layer_scores: Dict[int, float],
):
    """
    Generate a text report summarizing findings.
    """
    with open(results_file) as f:
        results = json.load(f)
    
    summary = results.get("summary", {})
    probes = results.get("probes", {})
    
    report = []
    report.append("=" * 70)
    report.append("COST PREDICTION POC - ANALYSIS REPORT")
    report.append("=" * 70)
    
    report.append("\n## SUMMARY\n")
    report.append(f"Baseline R² (input length only): {summary.get('baseline_r2', 'N/A'):.4f}")
    report.append(f"Best probe: {summary.get('best_probe', 'N/A')}")
    report.append(f"Best R²: {summary.get('best_r2', 'N/A'):.4f}")
    report.append(f"Improvement over baseline: {summary.get('improvement_over_baseline', 'N/A'):+.4f}")
    
    report.append("\n## LAYER ANALYSIS\n")
    if layer_scores:
        best_layer = max(layer_scores, key=layer_scores.get)
        report.append(f"Most predictive layer: {best_layer} (R² = {layer_scores[best_layer]:.4f})")
        report.append("\nAll layers:")
        for layer, score in sorted(layer_scores.items()):
            report.append(f"  Layer {layer}: R² = {score:.4f}")
    
    report.append("\n## SUCCESS CRITERIA\n")
    report.append(f"R² > 0.30 (meaningful): {'✓ PASSED' if summary.get('meets_meaningful_threshold') else '✗ FAILED'}")
    report.append(f"R² > 0.50 (useful): {'✓ PASSED' if summary.get('meets_useful_threshold') else '✗ FAILED'}")
    report.append(f"Activations add value: {'✓ YES' if summary.get('activation_adds_value') else '✗ NO'}")
    
    report.append("\n## RECOMMENDATIONS\n")
    best_r2 = summary.get('best_r2', 0)
    if best_r2 > 0.50:
        report.append("STRONG SIGNAL: Proceed with full system development.")
        report.append("- Build end-to-end budget-constrained inference prototype")
        report.append("- Test with real API traffic patterns")
        report.append("- Write paper draft")
    elif best_r2 > 0.30:
        report.append("MODERATE SIGNAL: Worth further investigation.")
        report.append("- Try different feature engineering approaches")
        report.append("- Test on more diverse prompt distributions")
        report.append("- Consider classification (short/medium/long) instead of regression")
    elif best_r2 > 0.15:
        report.append("WEAK SIGNAL: Activations show some predictive power.")
        report.append("- Investigate specific prompt types where prediction works")
        report.append("- Try non-linear probes (MLP)")
        report.append("- Consider if the task is inherently unpredictable")
    else:
        report.append("MINIMAL SIGNAL: Current approach may not be viable.")
        report.append("- Generation length appears hard to predict from prefill")
        report.append("- Consider alternative approaches (runtime monitoring)")
        report.append("- Document negative result for publication")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_dir / 'analysis_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Cost Prediction Analysis")
    parser.add_argument("--data-dir", type=str, default="./cost_prediction_data",
                        help="Directory with collected data")
    parser.add_argument("--results-dir", type=str, default="./cost_prediction_results",
                        help="Directory with probe results")
    parser.add_argument("--output-dir", type=str, default="./cost_prediction_analysis",
                        help="Output directory for plots and report")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    activations, features, targets, metadata = load_data(data_dir)
    print(f"Loaded {len(targets)} samples")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_predicted_vs_actual(
        activations, features, targets, output_dir,
        alpha=args.alpha, seed=args.seed
    )
    
    layer_scores = plot_layer_comparison(
        activations, targets, metadata, output_dir,
        alpha=args.alpha, seed=args.seed
    )
    
    plot_output_distribution(targets, metadata, output_dir)
    
    plot_input_vs_output(features, targets, output_dir)
    
    plot_error_analysis(
        activations, features, targets, metadata, output_dir,
        alpha=args.alpha, seed=args.seed
    )
    
    # Generate report
    results_file = results_dir / "probe_results.json"
    if results_file.exists():
        generate_report(results_file, output_dir, layer_scores)
    else:
        print(f"\nWarning: Results file not found at {results_file}")
        print("Run step2_train_probe.py first to generate results.")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
