import os
import json
import argparse
import subprocess
from pathlib import Path

def create_trial_config(trial_num, base_config, modifications):
    """Create a new trial configuration with modifications."""
    config = base_config.copy()
    config.update(modifications)
    config["output_dir"] = f"results/trial_{trial_num}"
    
    config_path = f"config_trial_{trial_num}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config_path

def run_trial(config_path, runpod_template_id):
    """Start a RunPod instance with the given configuration."""
    # Command to start a RunPod instance
    cmd = [
        "runpod",
        "pod",
        "create",
        "--template", runpod_template_id,
        "--gpu", "1",
        "--volume", "HAABSA_PLUS_PLUS_DA_LLMS:/workspace/HAABSA_PLUS_PLUS_DA_LLMS",
        "--command", f"cd /workspace/HAABSA_PLUS_PLUS_DA_LLMS && python runpod_entry.py --config {config_path}"
    ]
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run multiple HAABSA++ trials on RunPod")
    parser.add_argument("--template", type=str, required=True, help="RunPod template ID")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials to run")
    args = parser.parse_args()

    # Base configuration
    base_config = {
        "year": "2015",
        "model": "lcrModelAlt_hierarchical_v4",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50,
        "embedding_type": "bert",
        "cross_validation": True,
        "save_model": True
    }

    # Different configurations to try
    trial_configs = [
        {"batch_size": 32, "learning_rate": 0.001, "embedding_type": "bert"},
        {"batch_size": 64, "learning_rate": 0.0005, "embedding_type": "elmo"},
        {"batch_size": 128, "learning_rate": 0.0001, "embedding_type": "glove"},
        {"batch_size": 16, "learning_rate": 0.002, "embedding_type": "bert"},
        {"batch_size": 48, "learning_rate": 0.0008, "embedding_type": "elmo"}
    ]

    # Create and run trials
    for i in range(min(args.num_trials, len(trial_configs))):
        config_path = create_trial_config(i + 1, base_config, trial_configs[i])
        print(f"Starting trial {i + 1} with config: {config_path}")
        run_trial(config_path, args.template)

if __name__ == "__main__":
    main() 