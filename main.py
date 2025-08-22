
import argparse
from model_extraction_attack.config import load_config
from model_extraction_attack.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Model Extraction Refactored")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)

if __name__ == "__main__":
    main()
