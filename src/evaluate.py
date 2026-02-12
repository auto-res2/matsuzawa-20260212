"""
Evaluation script for comparing multiple runs.
Fetches metrics from WandB and generates comparison visualizations.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import wandb
import matplotlib.pyplot as plt
import numpy as np


def fetch_wandb_metrics(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch metrics for a single run from WandB.
    
    Returns:
        Dict with 'config', 'summary', and 'history'
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        return {
            "run_id": run_id,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "history": run.history().to_dict('records') if hasattr(run, 'history') else [],
        }
    except Exception as e:
        print(f"Warning: Failed to fetch metrics for {run_id}: {e}")
        return {
            "run_id": run_id,
            "config": {},
            "summary": {},
            "history": [],
        }


def load_local_results(results_dir: Path, run_id: str) -> Dict:
    """
    Load results from local file as fallback.
    """
    results_file = results_dir / run_id / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def evaluate_runs(results_dir: str, run_ids: List[str], entity: str, project: str) -> None:
    """
    Evaluate and compare multiple runs.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating {len(run_ids)} runs...")
    
    # Fetch metrics for each run
    all_metrics = {}
    for run_id in run_ids:
        print(f"  Fetching metrics for {run_id}...")
        
        # Try WandB first
        wandb_metrics = fetch_wandb_metrics(entity, project, run_id)
        
        # Load local results as fallback
        local_results = load_local_results(results_path, run_id)
        
        # Combine metrics
        metrics = {
            "run_id": run_id,
            "method": wandb_metrics["config"].get("run", {}).get("method", run_id),
            "accuracy": wandb_metrics["summary"].get("accuracy", 0.0),
            "correct": wandb_metrics["summary"].get("correct", 0),
            "total": wandb_metrics["summary"].get("total", 0),
            "verification_pass_rate": wandb_metrics["summary"].get("verification_pass_rate", 0.0),
            "config": wandb_metrics["config"],
            "local_results": local_results,
        }
        
        all_metrics[run_id] = metrics
        
        # Export per-run metrics
        run_dir = results_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"    Saved metrics to {metrics_file}")
    
    # Create comparison directory
    comparison_dir = results_path / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate metrics
    primary_metric = "accuracy"
    
    # Identify proposed vs baseline
    proposed_runs = [m for m in all_metrics.values() if "proposed" in m["run_id"]]
    baseline_runs = [m for m in all_metrics.values() if "comparative" in m["run_id"]]
    
    best_proposed = max(proposed_runs, key=lambda x: x[primary_metric]) if proposed_runs else None
    best_baseline = max(baseline_runs, key=lambda x: x[primary_metric]) if baseline_runs else None
    
    gap = 0.0
    if best_proposed and best_baseline:
        gap = best_proposed[primary_metric] - best_baseline[primary_metric]
    
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": {run_id: m[primary_metric] for run_id, m in all_metrics.items()},
        "best_proposed": {
            "run_id": best_proposed["run_id"],
            "accuracy": best_proposed["accuracy"],
            "verification_pass_rate": best_proposed["verification_pass_rate"],
        } if best_proposed else None,
        "best_baseline": {
            "run_id": best_baseline["run_id"],
            "accuracy": best_baseline["accuracy"],
            "verification_pass_rate": best_baseline["verification_pass_rate"],
        } if best_baseline else None,
        "gap": gap,
    }
    
    # Save aggregated metrics
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated metrics to {aggregated_file}")
    
    # Generate comparison figures
    generate_comparison_figures(all_metrics, comparison_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Comparison Summary:")
    print(f"  Primary metric: {primary_metric}")
    if best_proposed:
        print(f"  Best proposed: {best_proposed['run_id']} ({best_proposed[primary_metric]:.4f})")
    if best_baseline:
        print(f"  Best baseline: {best_baseline['run_id']} ({best_baseline[primary_metric]:.4f})")
    print(f"  Gap: {gap:+.4f}")
    print(f"{'='*80}")


def generate_comparison_figures(all_metrics: Dict, comparison_dir: Path) -> None:
    """
    Generate comparison visualizations.
    """
    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = list(all_metrics.keys())
    accuracies = [all_metrics[rid]["accuracy"] for rid in run_ids]
    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels([rid.replace("-", "\n") for rid in run_ids], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison: ET-CoT vs Baseline CoT")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    fig_path = comparison_dir / "accuracy_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Generated {fig_path}")
    
    # Figure 2: Verification pass rate (for ET-CoT runs)
    et_cot_runs = {rid: m for rid, m in all_metrics.items() if "proposed" in rid}
    if et_cot_runs:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        run_ids = list(et_cot_runs.keys())
        verification_rates = [et_cot_runs[rid]["verification_pass_rate"] for rid in run_ids]
        
        ax.bar(range(len(run_ids)), verification_rates, color="#e74c3c")
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels([rid.replace("-", "\n") for rid in run_ids], rotation=45, ha="right")
        ax.set_ylabel("Verification Pass Rate")
        ax.set_title("ET-CoT Verification Pass Rate")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        fig_path = comparison_dir / "verification_pass_rate.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Generated {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    parser.add_argument("--entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--project", type=str, default="2026-02-12", help="WandB project")
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    
    evaluate_runs(args.results_dir, run_ids, args.entity, args.project)


if __name__ == "__main__":
    main()
