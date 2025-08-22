"""Module for running model extraction experiments based on configurations."""

from pathlib import Path
import torch
from .utils import setup_logger, save_json, snapshot_env, set_seed
from .builder import (
    build_victim_model,
    build_student_model,
    build_oracle,
    build_query_loader,
    build_attack,
)

def run_experiment(config: dict):
    """Runs the model extraction experiment based on the provided configuration.

    This function sets up the environment, builds all necessary components (victim, student, oracle, data loaders, and attack),
    executes the attack, and saves the results.

    Args:
        config: A dictionary containing the experiment configuration.
    """
    # Setup
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config["logging"]["log"], config["logging"]["log_level"])
    snapshot_env(run_dir / "env.json")
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Build components
    logger.info("Building components...")
    victim = build_victim_model(config, device)
    oracle = build_oracle(config, victim)
    student = build_student_model(config, device)
    query_loader, test_loader = build_query_loader(config)

    img_shape = None
    if test_loader is not None and len(test_loader.dataset) > 0:
        img_shape = test_loader.dataset[0][0].shape
    elif query_loader is not None and len(query_loader.dataset) > 0:
        img_shape = query_loader.dataset[0][0].shape

    attack = build_attack(config, student, oracle, query_loader, device, test_loader=test_loader, img_shape=img_shape, run_dir=run_dir)

    # Run attack
    logger.info(f"Running attack: {config['attack']['name']}...")
    student, n_queries = attack.run()

    # Save results
    logger.info("Saving results...")
    torch.save(student.state_dict(), run_dir / "student.pth")
    results = {
        "attack": config["attack"],
        "n_queries": n_queries,
    }
    save_json(run_dir / "results.json", results)

    logger.info("Experiment finished.")