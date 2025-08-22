import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison_metrics(experiments: list, output_filename="comparison_metrics_plot.png"):
    """Visualizes and compares metrics from multiple attack experiments."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Model Extraction Attack Comparison', fontsize=16)

    for exp_config in experiments:
        metrics_file = exp_config['metrics_file']
        label = exp_config['label']
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
        except FileNotFoundError:
            print(f"Error: Metrics file '{metrics_file}' not found. Skipping {label}.")
            continue

        if not metrics_history:
            print(f"No metrics data to plot for {label}. Skipping.")
            continue

        # Extract data for plotting
        rounds = [d['round'] for d in metrics_history]
        n_queries = [d['n_queries'] for d in metrics_history]
        
        # Metric specific extraction
        if label == 'PRADA':
            agreement_metric = [d.get('test_agreement_f1_macro', np.nan) for d in metrics_history]
            # Papernot transferability metrics (if available)
            transferability_untargeted = [d['papernot_transferability'].get('transferability_untargeted', np.nan) if 'papernot_transferability' in d else np.nan for d in metrics_history]
            transferability_targeted = [d['papernot_transferability'].get('transferability_targeted', np.nan) if 'papernot_transferability' in d else np.nan for d in metrics_history]
            rnd_agreement = [d['papernot_transferability'].get('rnd_agreement', np.nan) if 'papernot_transferability' in d else np.nan for d in metrics_history]
            # RU Agreement for PRADA
            ru_agreement_accuracy = [d.get('ru_agreement_accuracy', np.nan) for d in metrics_history]
        else: # Tramer and Knockoff
            agreement_metric = [d.get('agreement_score', np.nan) for d in metrics_history]
            transferability_untargeted = [np.nan] * len(rounds) # Not directly available for Tramer/Knockoff in this format
            transferability_targeted = [np.nan] * len(rounds)
            rnd_agreement = [np.nan] * len(rounds)
            ru_agreement_accuracy = [np.nan] * len(rounds) # Not directly available for Tramer/Knockoff in this format

        # Current x_pool size (if available)
        current_labeled_dataset_size = [d.get('current_labeled_dataset_size', np.nan) for d in metrics_history]

        # Plot 1: Agreement Metric
        axes[0].plot(rounds, agreement_metric, marker='o', linestyle='-', label=f'{label} Agreement')
        if label == 'PRADA': # Plot RU Agreement for PRADA only
            axes[0].plot(rounds, ru_agreement_accuracy, marker='x', linestyle='--', label=f'{label} RU Agreement')
        
        # Plot 2: Papernot Transferability Metrics (only for PRADA)
        if label == 'PRADA':
            axes[1].plot(rounds, transferability_untargeted, marker='o', linestyle='-', label=f'{label} Transferability (Untargeted)')
            axes[1].plot(rounds, transferability_targeted, marker='x', linestyle='--', label=f'{label} Transferability (Targeted)')
            axes[1].plot(rounds, rnd_agreement, marker='s', linestyle=':', label=f'{label} RND Agreement')

        # Plot 3: Labeled Dataset Size Growth
        axes[2].plot(rounds, current_labeled_dataset_size, marker='o', linestyle='-', label=f'{label} Labeled Dataset Size')

    # Set common labels and titles for Plot 1
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Student Model Agreement')
    axes[0].legend()
    axes[0].grid(True)

    # Set common labels and titles for Plot 2
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Papernot Transferability Metrics (PRADA Only)')
    axes[1].legend()
    axes[1].grid(True)

    # Set common labels and titles for Plot 3
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Size')
    axes[2].set_title('Labeled Dataset Size Growth')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot
    output_dir = "runs/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    experiments_to_compare = [
        {'metrics_file': 'runs/prada_mnist_experiment/prada_metrics.json', 'label': 'PRADA'},
        {'metrics_file': 'runs/tramer_mnist_experiment/tramer_metrics.json', 'label': 'Tramer'},
    ]
    plot_comparison_metrics(experiments_to_compare)
