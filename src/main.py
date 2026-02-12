"""
Main orchestrator for inference-only prompt tuning experiments.
Loads config and invokes inference.py as a subprocess.
"""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single run_id for inference-only prompt tuning.
    """
    # Apply mode overrides
    if cfg.mode == "sanity_check":
        # Reduce dataset size for sanity check
        cfg.dataset.max_samples = 10
        # Use sanity WandB namespace
        if "sanity" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        # Lower temperature for more consistent formatting in sanity checks
        # For structured output formats like ET-CoT, use very low temperature
        if cfg.run.method == "et_cot":
            cfg.model.temperature = 0.1
            cfg.model.do_sample = True
        elif cfg.model.temperature > 0.3:
            cfg.model.temperature = 0.3
    
    # Print configuration
    print("=" * 80)
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method}")
    print(f"Mode: {cfg.mode}")
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name} ({cfg.dataset.max_samples} samples)")
    print("=" * 80)
    
    # Save resolved config
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Saved config to {config_path}")
    
    # Invoke inference.py as subprocess
    print("\nStarting inference...")
    inference_script = Path(__file__).parent / "inference.py"
    
    cmd = [
        sys.executable,
        "-u",
        str(inference_script),
        f"--config={config_path}",
    ]
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\nInference failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\nRun {cfg.run.run_id} completed successfully!")


if __name__ == "__main__":
    main()
