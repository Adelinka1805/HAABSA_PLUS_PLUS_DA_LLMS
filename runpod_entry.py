import os
import argparse
import subprocess
import json
import tensorflow as tf
from pathlib import Path

def download_embeddings(year):
    """Download required embeddings for the specified year."""
    base_url = "https://drive.google.com/uc?id="
    embeddings = {
        "2015": {
            "glove": "14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu",
            "elmo": "1GfHKLmbiBEkATkeNmJq7CyXGo61aoY2l",
            "bert": "1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx"
        },
        "2016": {
            "glove": "1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92",
            "elmo": "1OT_1p55LNc4vxc0IZksSj2PmFraUIlRD",
            "bert": "1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2"
        }
    }
    
    for embedding_type, file_id in embeddings[year].items():
        output_file = f"data/embeddings/{year}_{embedding_type}.zip"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        subprocess.run([
            "gdown",
            f"{base_url}{file_id}",
            "-O", output_file
        ])

def prepare_data(year):
    """Prepare data for the specified year."""
    # Run BERT preparation
    subprocess.run(["python", "prepare_bert.py"])
    
    # Run ELMo preparation
    subprocess.run(["python", "prepareELMo.py"])

def run_training(config):
    """Run the training process with the given configuration."""
    # Import config first to get all flags
    import config
    
    # Define required flags and their types
    required_flags = {
        "embedding_type": str,
        "year": int,
        "batch_size": int,
        "learning_rate": float,
        "epochs": int,
        "da_type": str,
        "embedding_dim": int,
        "n_hidden": int,
        "keep_prob1": float,
        "keep_prob2": float,
        "momentum": float,
        "l2_reg": float
    }
    
    # Validate configuration
    for flag_name, flag_type in required_flags.items():
        if flag_name not in config:
            raise ValueError(f"Missing required configuration: {flag_name}")
        try:
            # Convert to correct type
            config[flag_name] = flag_type(config[flag_name])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {flag_name}: {str(e)}")
    
    # Set default values for optional flags
    default_flags = {
        "da_type": "none",
        "embedding_dim": 768,
        "n_hidden": 300,
        "keep_prob1": 0.6,
        "keep_prob2": 0.6,
        "momentum": 0.9,
        "l2_reg": 0.01
    }
    
    # Update flags with configuration values
    try:
        # Required flags
        config.FLAGS.embedding_type = config["embedding_type"]
        config.FLAGS.year = config["year"]
        config.FLAGS.batch_size = config["batch_size"]
        config.FLAGS.learning_rate = config["learning_rate"]
        config.FLAGS.n_iter = config["epochs"]
        
        # Optional flags with defaults
        for flag_name, default_value in default_flags.items():
            setattr(config.FLAGS, flag_name, config.get(flag_name, default_value))
            
        # Additional flags that might be needed
        if "max_sentence_len" in config:
            config.FLAGS.max_sentence_len = config["max_sentence_len"]
        if "max_target_len" in config:
            config.FLAGS.max_target_len = config["max_target_len"]
        if "n_class" in config:
            config.FLAGS.n_class = config["n_class"]
            
    except AttributeError as e:
        raise ValueError(f"Failed to set flag: {str(e)}")
    
    # Now import and run main
    import main
    main.main(None)

def main():
    parser = argparse.ArgumentParser(description="Run HAABSA++ training on RunPod")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create necessary directories
    os.makedirs("data/programGeneratedData", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Download embeddings
    download_embeddings(config["year"])

    # Prepare data
    prepare_data(config["year"])

    # Run training
    run_training(config)

if __name__ == "__main__":
    main() 