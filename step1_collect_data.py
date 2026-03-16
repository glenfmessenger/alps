"""
Cost Prediction PoC v2 - Clean Experiment

Key changes from v1:
1. Uses curated prompts designed to produce varied-length responses
2. Higher max_tokens (4096) to avoid truncation
3. Tracks prompt category for analysis
4. Filters out truncated responses for clean evaluation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import time

from prompts_curated import get_balanced_sample, get_all_prompts, CuratedPrompt


@dataclass
class PromptResult:
    """Result from processing a single prompt."""
    prompt: str
    prompt_tokens: int
    output_tokens: int
    output_text: str
    stop_reason: str  # "eos", "max_tokens"
    latency_ms: float
    category: str  # "short", "medium", "long"
    subcategory: str
    activations: Dict[str, np.ndarray]


class CostDataCollector:
    """Collects prefill activations and generation lengths."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        layer_fractions: List[float] = [0.25, 0.5, 0.75, 1.0],
    ):
        self.model_name = model_name
        self.device = device
        self.layer_fractions = layer_fractions
        
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        
        self.layer_indices = [
            min(int(f * self.num_layers), self.num_layers - 1)
            for f in layer_fractions
        ]
        self.layer_names = [f"layer_{idx}" for idx in self.layer_indices]
        
        print(f"Model: {self.num_layers} layers, {self.hidden_dim} hidden dim")
        print(f"Extracting from layers: {self.layer_indices}")
        
        self._activations: Dict[int, np.ndarray] = {}
        self._hooks: List[Any] = []
    
    def _setup_hooks(self):
        """Register forward hooks for activation extraction."""
        self._activations = {}
        self._hooks = []
        
        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]
            
            def make_hook(idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Last token position
                    act = hidden[0, -1, :].detach().float().cpu().numpy()
                    self._activations[idx] = act
                return hook_fn
            
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}
    
    def process_prompt(
        self,
        prompt: CuratedPrompt,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Optional[PromptResult]:
        """Process a single prompt."""
        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt.text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            prompt_tokens = inputs.input_ids.shape[1]
            
            self._setup_hooks()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            activations = {
                f"layer_{idx}": self._activations[idx]
                for idx in self.layer_indices
                if idx in self._activations
            }
            
            self._remove_hooks()
            
            total_tokens = outputs.shape[1]
            output_tokens = total_tokens - prompt_tokens
            
            output_text = self.tokenizer.decode(
                outputs[0, prompt_tokens:],
                skip_special_tokens=True
            )
            
            last_token = outputs[0, -1].item()
            if last_token == self.tokenizer.eos_token_id:
                stop_reason = "eos"
            elif output_tokens >= max_new_tokens:
                stop_reason = "max_tokens"
            else:
                stop_reason = "unknown"
            
            return PromptResult(
                prompt=prompt.text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                output_text=output_text,
                stop_reason=stop_reason,
                latency_ms=latency_ms,
                category=prompt.category,
                subcategory=prompt.subcategory,
                activations=activations,
            )
            
        except Exception as e:
            self._remove_hooks()
            print(f"\nError: {e}")
            return None
    
    def collect_dataset(
        self,
        prompts: List[CuratedPrompt],
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Process all prompts and return training data."""
        results: List[PromptResult] = []
        
        for prompt in tqdm(prompts, desc="Collecting data"):
            result = self.process_prompt(prompt, max_new_tokens, temperature)
            if result is not None and len(result.activations) == len(self.layer_indices):
                results.append(result)
        
        print(f"\nProcessed {len(results)}/{len(prompts)} prompts")
        
        # Count EOS vs max_tokens
        eos_count = sum(1 for r in results if r.stop_reason == "eos")
        max_count = sum(1 for r in results if r.stop_reason == "max_tokens")
        print(f"EOS terminations: {eos_count} ({100*eos_count/len(results):.1f}%)")
        print(f"Max tokens hit: {max_count} ({100*max_count/len(results):.1f}%)")
        
        # Convert to arrays
        n_prompts = len(results)
        n_layers = len(self.layer_indices)
        
        activations = np.zeros((n_prompts, n_layers, self.hidden_dim))
        for i, result in enumerate(results):
            for j, layer_name in enumerate(self.layer_names):
                activations[i, j, :] = result.activations[layer_name]
        
        features = self._compute_features(activations, results)
        targets = np.array([r.output_tokens for r in results])
        
        metadata = {
            "n_prompts": n_prompts,
            "n_layers": n_layers,
            "hidden_dim": self.hidden_dim,
            "layer_indices": self.layer_indices,
            "layer_names": self.layer_names,
            "model_name": self.model_name,
            "prompts": [r.prompt for r in results],
            "prompt_tokens": [r.prompt_tokens for r in results],
            "output_tokens": [r.output_tokens for r in results],
            "stop_reasons": [r.stop_reason for r in results],
            "categories": [r.category for r in results],
            "subcategories": [r.subcategory for r in results],
            "latencies_ms": [r.latency_ms for r in results],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        
        return activations, features, targets, metadata
    
    def _compute_features(
        self,
        activations: np.ndarray,
        results: List[PromptResult],
    ) -> np.ndarray:
        """Compute derived features."""
        n_prompts = activations.shape[0]
        n_layers = activations.shape[1]
        
        n_features = n_layers * 5 + 2
        features = np.zeros((n_prompts, n_features))
        
        for i in range(n_prompts):
            feat_idx = 0
            for j in range(n_layers):
                act = activations[i, j, :]
                features[i, feat_idx] = np.linalg.norm(act)
                features[i, feat_idx + 1] = act.mean()
                features[i, feat_idx + 2] = act.std()
                features[i, feat_idx + 3] = act.max()
                features[i, feat_idx + 4] = act.min()
                feat_idx += 5
            
            features[i, feat_idx] = results[i].prompt_tokens
            features[i, feat_idx + 1] = np.log1p(results[i].prompt_tokens)
        
        return features


def main():
    parser = argparse.ArgumentParser(description="Cost Prediction Data Collection v2")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts (will be balanced across categories)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max generation length (high to avoid truncation)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="./cost_prediction_v2_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get balanced prompts
    print(f"Loading {args.n_prompts} balanced prompts...")
    prompts = get_balanced_sample(args.n_prompts, seed=args.seed)
    
    # Print category distribution
    cat_counts = {}
    for p in prompts:
        cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
    print(f"Category distribution: {cat_counts}")
    
    # Collect data
    collector = CostDataCollector(
        model_name=args.model,
        device=args.device,
    )
    
    activations, features, targets, metadata = collector.collect_dataset(
        prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Save all data
    np.save(output_dir / "activations.npy", activations)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "targets.npy", targets)
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Also save EOS-only subset for clean analysis
    eos_mask = np.array([r == "eos" for r in metadata["stop_reasons"]])
    if eos_mask.sum() > 10:
        np.save(output_dir / "activations_eos.npy", activations[eos_mask])
        np.save(output_dir / "features_eos.npy", features[eos_mask])
        np.save(output_dir / "targets_eos.npy", targets[eos_mask])
        
        eos_metadata = {
            "n_prompts": int(eos_mask.sum()),
            "n_layers": metadata["n_layers"],
            "hidden_dim": metadata["hidden_dim"],
            "layer_indices": metadata["layer_indices"],
            "layer_names": metadata["layer_names"],
            "model_name": metadata["model_name"],
            "prompts": [p for p, m in zip(metadata["prompts"], eos_mask) if m],
            "prompt_tokens": [p for p, m in zip(metadata["prompt_tokens"], eos_mask) if m],
            "output_tokens": [p for p, m in zip(metadata["output_tokens"], eos_mask) if m],
            "stop_reasons": [p for p, m in zip(metadata["stop_reasons"], eos_mask) if m],
            "categories": [p for p, m in zip(metadata["categories"], eos_mask) if m],
            "subcategories": [p for p, m in zip(metadata["subcategories"], eos_mask) if m],
        }
        
        with open(output_dir / "metadata_eos.json", "w") as f:
            json.dump(eos_metadata, f, indent=2)
        
        print(f"\nEOS-only subset saved: {eos_mask.sum()} samples")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Total samples: {len(targets)}")
    print(f"EOS terminations: {eos_mask.sum()} ({100*eos_mask.sum()/len(targets):.1f}%)")
    
    print(f"\nOutput length statistics (ALL):")
    print(f"  Min: {targets.min()}")
    print(f"  Max: {targets.max()}")
    print(f"  Mean: {targets.mean():.1f}")
    print(f"  Std: {targets.std():.1f}")
    
    if eos_mask.sum() > 0:
        eos_targets = targets[eos_mask]
        print(f"\nOutput length statistics (EOS only):")
        print(f"  Min: {eos_targets.min()}")
        print(f"  Max: {eos_targets.max()}")
        print(f"  Mean: {eos_targets.mean():.1f}")
        print(f"  Std: {eos_targets.std():.1f}")
    
    # By category
    print(f"\nBy category:")
    for cat in ["short", "medium", "long"]:
        cat_mask = np.array([c == cat for c in metadata["categories"]])
        if cat_mask.sum() > 0:
            cat_targets = targets[cat_mask]
            cat_eos = sum(1 for c, r in zip(metadata["categories"], metadata["stop_reasons"]) 
                        if c == cat and r == "eos")
            print(f"  {cat}: n={cat_mask.sum()}, mean={cat_targets.mean():.0f}, "
                  f"range=[{cat_targets.min()}-{cat_targets.max()}], "
                  f"EOS={cat_eos}/{cat_mask.sum()}")


if __name__ == "__main__":
    main()
