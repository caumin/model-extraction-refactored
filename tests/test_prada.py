import pytest
import torch
from pathlib import Path

from model_extraction_attack.config import load_config
from model_extraction_attack.builder import build_victim_model, build_student_model, build_oracle, build_query_loader, build_attack

@pytest.fixture(scope="module")
def prada_config():
    # Create a minimal config for testing Prada
    config = {
        "run_dir": "runs/test_prada",
        "device": "cpu",
        "seed": 42,
        "victim": {
            "arch": "prada_mnist_target",
            "ckpt": "./ckpts/prada_mnist_target.pt",
            "num_classes": 10,
            "in_channels": 1
        },
        "student": {
            "arch": "resnet18"
        },
        "attack": {
            "name": "prada",
            "label_only": True,
            "prada": {
                "rounds": 1,
                "lambda_aug": 0.1,
                "reservoir": 10,
                "epochs": 1,
                "batch_size": 10,
                "lr": 1e-3,
                "crafter": "jbda",
                "target_policy": "next"
            }
        },
        "query": {
            "dataset": "mnist",
            "data_dir": "./data",
            "batch_size": 10,
            "img_size": 28
        },
        "logging": {
            "log": False,
            "log_level": "DEBUG"
        }
    }
    return config

def test_prada_attack_runs(prada_config):
    config = prada_config
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])

    victim = build_victim_model(config, device, run_dir)
    oracle = build_oracle(config, victim)
    student = build_student_model(config, device)
    query_loader = build_query_loader(config)

    attack = build_attack(config, student, oracle, query_loader, device)

    final_student, n_queries = attack.run()

    assert isinstance(final_student, torch.nn.Module)
    assert n_queries > 0
