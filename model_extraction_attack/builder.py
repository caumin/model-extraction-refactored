"""Module for building various components of the model extraction experiment."""

from pathlib import Path
import torch
from .models import build_model, load_checkpoint
from .oracles import Oracle
from .data import get_mnist_loaders, get_imagefolder_loaders, get_unlabeled_loader, get_cifar10_loaders
from torch.utils.data import DataLoader
from .attacks.registry import ATTACK_REGISTRY
from typing import Optional, Tuple

# Import all attack modules to ensure they are registered
from .attacks import papernot, knockoff, tramer, prada, dfme, cloud_leak, maze


def build_victim_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Builds and loads the victim model based on the configuration.

    Args:
        config: The experiment configuration dictionary.
        device: The device to load the model onto.

    Returns:
        The loaded victim model.
    """
    victim_config = config["victim"]
    model = build_model(
        victim_config["arch"],
        num_classes=victim_config["num_classes"],
        in_channels=victim_config["in_channels"],
        device=device,
    )
    # The path should be relative to the project root or absolute.
    ckpt_path = Path(victim_config["ckpt"])
    load_checkpoint(model, str(ckpt_path), map_location=device)
    return model


def build_student_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Builds the student model based on the configuration.

    Args:
        config: The experiment configuration dictionary.
        device: The device to load the model onto.

    Returns:
        The initialized student model.
    """
    student_config = config["student"]
    victim_config = config["victim"]
    
    # Get dropout_rate from student config, default to 0.0 if not specified
    dropout_rate = student_config.get("dropout_rate", 0.0)

    model = build_model(
        student_config["arch"],
        num_classes=victim_config["num_classes"],
        in_channels=victim_config["in_channels"],
        device=device,
        dropout_rate=dropout_rate, # Pass dropout_rate
    )
    return model


def build_oracle(config: dict, model: torch.nn.Module) -> Oracle:
    """Builds the oracle for querying the victim model.

    Args:
        config: The experiment configuration dictionary.
        model: The victim model to be used as the oracle.

    Returns:
        An Oracle instance.
    """
    attack_config = config["attack"]
    disclosure = "label" if attack_config["label_only"] else "probs"
    return Oracle(model, disclosure=disclosure)


def build_query_loader(config: dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Builds the data loader for querying the oracle.

    Args:
        config: The experiment configuration dictionary.

    Returns:
        A tuple containing the DataLoader instance for the query dataset and an optional test DataLoader.
    """
    query_config = config["query"]
    dataset = query_config["dataset"]
    data_dir = query_config["data_dir"]
    batch_size = query_config["batch_size"]
    img_size = query_config["img_size"]
    in_channels = config["victim"]["in_channels"]

    if dataset == "mnist":
        train_loader, test_loader, _ = get_mnist_loaders(batch_size, img_size, root=data_dir, in_ch=in_channels)
        return train_loader, test_loader
    elif dataset == "cifar10":
        train_loader, test_loader, _ = get_cifar10_loaders(batch_size, img_size, root=data_dir, in_ch=in_channels)
        return train_loader, test_loader
    elif dataset == "imagefolder":
        train_loader, test_loader, _ = get_imagefolder_loaders(data_dir, batch_size, img_size, in_ch=in_channels)
        return train_loader, test_loader
    elif dataset == "images":
        return get_unlabeled_loader(data_dir, batch_size, img_size, in_ch=in_channels), None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_attack(config: dict, student: torch.nn.Module, oracle: Oracle, query_loader: DataLoader, device: torch.device, test_loader: Optional[DataLoader] = None, img_shape: Optional[Tuple[int, ...]] = None, run_dir: Optional[Path] = None):
    """Builds and initializes the specified attack using the registry.

    Args:
        config: The experiment configuration dictionary.
        student: The student model.
        oracle: The oracle model.
        query_loader: The data loader for queries.
        device: The device to run the attack on.
        test_loader: Optional test data loader for evaluation.
        img_shape: Optional image shape for generating random samples.
        run_dir: Optional run directory for saving attack-specific outputs.

    Returns:
        An instance of the specified attack class.

    Raises:
        ValueError: If an unknown attack name is specified.
    """
    attack_name = config["attack"]["name"]
    attack_config = config["attack"].get(attack_name, {})

    try:
        attack_class = ATTACK_REGISTRY[attack_name]
    except KeyError:
        raise ValueError(f"Unknown attack: {attack_name}")

    # Generic arguments for all attacks
    kwargs = {
        "student": student,
        "oracle": oracle,
        "query_loader": query_loader,
        "device": device,
        "run_dir": run_dir, # Pass run_dir to all attacks
    }
    
    # Special handling for prada which has a different signature
    if attack_name == "prada":
        # Pass additional parameters for PRADA
        kwargs["test_loader"] = test_loader
        kwargs["img_shape"] = img_shape
        kwargs["query_budget"] = config["attack"].get("query_budget", None)
        kwargs["optimizer_type"] = attack_config.get("optimizer_type", "adam")
        kwargs["optimizer_params"] = attack_config.get("optimizer_params", {})
        # label_only and other prada-specific params are handled within PradaAttack's prada_config
        return attack_class(prada_config=attack_config, **kwargs)
    else:
        kwargs["label_only"] = config["attack"].get("label_only", False)
        kwargs.update(attack_config)

    return attack_class(**kwargs)
