# Developer Guide: Model Extraction Attack Repository

This guide provides an overview of the repository's structure, the workflow for implementing and running model extraction attacks, and guidelines for consistent development.

## 1. Repository Structure

```
.
├── main.py                     # Main entry point for running experiments
├── requirements.txt            # Python dependencies
├── configs/                    # Experiment configuration files (YAML)
│   ├── default.yaml            # Default configuration
│   ├── prada_mnist_experiment.yaml # Example PRADA MNIST config
│   ├── tramer_mnist_experiment.yaml # Example Tramer MNIST config
│   └── knockoff_mnist_experiment.yaml # Example Knockoff MNIST config
├── ckpts/                      # Pre-trained model checkpoints
├── data/                       # Datasets (e.g., MNIST, CIFAR10)
├── runs/                       # Experiment output directories (logs, metrics, trained models)
│   ├── prada_mnist_experiment/
│   ├── tramer_mnist_experiment/
│   └── ...
├── model_extraction_attack/    # Main Python package for the project
│   ├── __init__.py
│   ├── builder.py              # Builds models, data loaders, and attack instances
│   ├── config.py               # Handles configuration loading and parsing
│   ├── crafter.py              # Utility functions for generating synthetic samples (e.g., adversarial examples)
│   ├── data.py                 # Data loading and preprocessing utilities
│   ├── metrics.py              # Functions for calculating various evaluation metrics
│   ├── models.py               # Neural network model definitions (target and substitute)
│   ├── oracles.py              # Defines the black-box oracle interface for querying victim models
│   ├── runner.py               # Orchestrates the experiment execution
│   ├── utils.py                # General utility functions (e.g., logging, seeding, soft_cross_entropy)
│   └── attacks/                # Implementations of different model extraction attacks
│       ├── __init__.py         # Registers attack classes
│       ├── base_attack.py      # Base class for all attacks
│       ├── prada.py            # PRADA attack implementation
│       ├── tramer.py           # Tramer attack implementation
│       ├── knockoff.py         # Knockoff attack implementation
│       ├── ...                 # Other attack implementations
│       └── registry.py         # Central registry for attack classes
└── tests/                      # Unit and integration tests
```

## 2. Workflow for Implementing and Running Attacks

### 2.1 Adding a New Attack

To add a new model extraction attack:

1.  **Create a new attack file:** In `model_extraction_attack/attacks/`, create a new Python file (e.g., `my_new_attack.py`).
2.  **Implement the Attack Class:**
    *   Your attack class should inherit from `model_extraction_attack.attacks.base_attack.BaseAttack`.
    *   Implement the `__init__` method to accept necessary parameters (student model, oracle, query loader, device, and any attack-specific configurations).
    *   Implement the `run()` method, which contains the core logic of your attack. This method should return the trained student model and the total number of queries made.
    *   Ensure your attack logs per-round metrics to `self.metrics_history` (a list of dictionaries) and saves it to a JSON file within `self.run_dir` at the end of the `run()` method. This is crucial for consistent visualization.
3.  **Register the Attack:**
    *   Add `@register_attack("my_new_attack_name")` decorator above your attack class definition.
    *   Import your new attack module in `model_extraction_attack/attacks/__init__.py` (e.g., `from . import my_new_attack`). This ensures your attack is registered and discoverable by the `builder`.
4.  **Create a Configuration File:**
    *   In `configs/`, create a new YAML file (e.g., `my_new_attack_experiment.yaml`).
    *   Define all necessary parameters for your attack, including `name: my_new_attack_name` under the `attack` section.
    *   Specify victim, student, and query data configurations.
5.  **Update `builder.py` (if necessary):**
    *   If your attack requires special handling or additional parameters not covered by the generic `kwargs` in `build_attack`, you might need to add a specific `if attack_name == "my_new_attack_name":` block in `model_extraction_attack/builder.py`. However, try to make your attack's `__init__` flexible enough to accept parameters via `**kwargs` to minimize changes here.
6.  **Add Tests:** Write unit and integration tests for your new attack in the `tests/` directory.

### 2.2 Running an Experiment

To run an experiment with a specific configuration:

```bash
python main.py --config configs/your_experiment_config.yaml
```

The results (logs, metrics JSON, trained student model checkpoint) will be saved in the `runs/your_experiment_name/` directory as specified in your configuration file.

### 2.3 Visualizing Results

To visualize and compare results from multiple experiments:

1.  **Ensure metrics JSON files are generated:** Run all desired experiments first.
2.  **Update `prada_viz.py`:** Modify the `experiments_to_compare` list in `prada_viz.py` to include the `metrics_file` path and a `label` for each experiment you want to compare.
3.  **Run the visualization script:**

    ```bash
    python prada_viz.py
    ```

    The comparison plot will be saved in `runs/comparison_plots/`.

## 3. Consistency Guidelines

To ensure consistency and maintainability across different attack implementations:

*   **BaseAttack Inheritance:** All attack classes must inherit from `model_extraction_attack.attacks.base_attack.BaseAttack`.
*   **`run()` Method Signature:** The `run()` method in your attack class should always return `Tuple[nn.Module, int]` (trained student model, total queries made).
*   **Configuration-Driven:** All attack-specific parameters should be defined in the YAML configuration files and accessed via the `prada_config` (or `attack_config` for non-PRADA attacks) dictionary passed to the `__init__` method.
*   **Metric Logging:** Log per-round metrics to `self.metrics_history` (a list of dictionaries) and save this list to a JSON file (e.g., `your_attack_metrics.json`) within `self.run_dir` at the end of the `run()` method. Each dictionary in `metrics_history` should include at least `round`, `n_queries`, and `current_labeled_dataset_size`. Include other relevant metrics as needed.
*   **Optimizer and Loss Function:** Use the `optimizer_type`, `optimizer_params`, `lr`, `epochs`, and `label_only` parameters from the configuration to set up the student model's training loop consistently. Leverage `soft_cross_entropy` from `utils.py` for soft labels.
*   **Data Handling:** Utilize `DataLoader`s for batching. For initial seed samples, use `make_seed_from_labeled` from `data.py`.
*   **Model Building:** Use `build_model` from `models.py` to instantiate student and victim models.

## 4. Reusable Utilities

The `model_extraction_attack/` package provides several utility modules that should be leveraged:

*   **`model_extraction_attack/metrics.py`:**
    *   `calculate_papernot_transferability`: For Papernot-style transferability metrics.
    *   `test_agreement`: Calculates macro-averaged F1-score on a test set.
    *   `ru_agreement`: Calculates accuracy on random uniform samples.
    *   `agreement`: General agreement between two models.
    *   **Usage:** Import and call these functions directly in your attack's `run()` method or helper functions.
*   **`model_extraction_attack/models.py`:**
    *   `build_model`: Factory function to create various neural network architectures (e.g., `prada_mnist_target`, `prada_sub_cnn2`).
    *   `load_checkpoint`: For loading pre-trained model weights.
    *   **Usage:** Call `build_model` in `builder.py` to instantiate models.
*   **`model_extraction_attack/crafter.py`:**
    *   Contains functions for generating synthetic samples (e.g., `fgsm_family_crafter`, `color_aug_batch`, `jsma_batch`).
    *   **Usage:** Import and call these functions in your attack's sample generation logic (e.g., `_create_new_samples`).
*   **`model_extraction_attack/data.py`:**
    *   `get_mnist_loaders`, `get_cifar10_loaders`, `get_imagefolder_loaders`: For loading standard datasets.
    *   `make_seed_from_labeled`: Crucial for creating initial seed datasets from labeled data.
    *   **Usage:** Data loaders are typically built in `builder.py` and passed to the attack. `make_seed_from_labeled` can be used within your attack's `run()` method for initial data setup.
*   **`model_extraction_attack/utils.py`:**
    *   `set_seed`: For reproducibility.
    *   `setup_logger`: For consistent logging.
    *   `save_json`: For saving results to JSON files.
    *   `soft_cross_entropy`: Centralized function for calculating soft cross-entropy loss.
    *   **Usage:** Import and use these general utility functions as needed.

---
