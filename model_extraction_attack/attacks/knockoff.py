from typing import Tuple, Dict, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .base_attack import BaseAttack
from ..metrics import agreement
from .registry import register_attack
from pathlib import Path # Added import for Path
from ..utils import soft_cross_entropy # Import centralized soft_cross_entropy

@register_attack("knockoff")
class KnockoffAttack(BaseAttack):
    """Implementation of the Knockoff Nets model extraction attack.

    This attack trains a substitute model by adaptively querying the target oracle
    based on a reward function.
    """
    def __init__(self, student: nn.Module, oracle, query_loader: DataLoader, device: torch.device,
                 query_budget: int, epochs: int, batch_size: int, lr: float, policy: str, reward: str,
                 temperature: float, policy_lr: float, label_only: bool, queries_per_round: int, div_subset_size: int,
                 run_dir: Optional[Path] = None):
        """Initializes the KnockoffAttack.

        Args:
            student: The student model to be trained.
            oracle: The black-box oracle model.
            query_loader: DataLoader for the initial query dataset.
            device: The device to run the attack on (e.g., 'cpu' or 'cuda').
            query_budget: Total number of queries allowed.
            epochs: Number of epochs to train the student in each round.
            batch_size: Batch size for student training.
            lr: Learning rate for student training.
            policy: Policy for selecting queries ('random' or 'adaptive').
            reward: Reward function for adaptive policy ('loss', 'cert', or 'div').
            temperature: Temperature parameter for softmax in adaptive policy.
            policy_lr: Learning rate for policy updates in adaptive policy.
            label_only: If True, oracle provides only labels; otherwise, probabilities.
            queries_per_round: Number of queries to make in each round.
            div_subset_size: Subset size for diversity reward calculation.
            run_dir: Optional run directory for saving attack-specific outputs.
        """
        super().__init__(student, oracle, query_loader, device)
        self.query_budget = query_budget
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.policy = policy
        self.reward = reward
        self.temperature = temperature
        self.policy_lr = policy_lr
        self.label_only = label_only
        self.queries_per_round = queries_per_round
        self.div_subset_size = div_subset_size
        self.metrics_history = [] # Initialize for logging metrics
        self.run_dir = run_dir # Store run_dir

    def run(self) -> Tuple[nn.Module, int]:
        """Executes the Knockoff Nets model extraction attack.

        Returns:
            A tuple containing the trained student model and the total number of queries made.
        """
        arms = self._get_initial_arms()
        classes = sorted(arms.keys())
        theta = torch.zeros(len(classes), dtype=torch.float)
        labeled_dataset = None
        queried_samples = torch.empty(0, dtype=torch.float)

        round_num = 0 # Track rounds for logging
        while self.n_queries < self.query_budget:
            queries_to_make = min(self.queries_per_round, self.query_budget - self.n_queries)
            if queries_to_make <= 0:
                break

            xb, yb_oracle, labeled_dataset, queried_samples = self._query_and_label(
                arms, classes, theta, queries_to_make, labeled_dataset, queried_samples
            )

            if xb is None:
                break

            train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True)
            self._train_student(train_loader)

            if self.policy == 'adaptive':
                theta = self._update_theta(xb, yb_oracle, classes, theta, queried_samples)

            # Calculate agreement metrics
            if labeled_dataset is not None:
                current_train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=False) # No shuffle for metrics
                agreement_score = agreement(self.student, self.oracle.model, current_train_loader, self.device)
                
                # Log metrics
                round_metrics = {
                    "round": round_num,
                    "n_queries": self.n_queries,
                    "agreement_score": agreement_score,
                    "current_labeled_dataset_size": len(labeled_dataset),
                }
                self.metrics_history.append(round_metrics)
                print(f"Round {round_num} metrics: {round_metrics}")
                round_num += 1

        # Save metrics history to a JSON file
        if self.metrics_history:
            import json
            from pathlib import Path
            output_path = self.run_dir / "knockoff_metrics.json" # Use run_dir
            with open(output_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            print(f"Metrics history saved to {output_path}")

        return self.student, self.n_queries

    def _get_initial_arms(self) -> Dict[int, torch.Tensor]:
        """Collects initial samples for each class to form the arms of the multi-armed bandit."""
        arms = {}
        for batch in self.query_loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            yb = self._query_oracle(xb.to(self.device)).cpu()
            for cls in yb.unique().tolist():
                idx = (yb == cls).nonzero(as_tuple=False).squeeze(1)
                if cls not in arms:
                    arms[cls] = []
                arms[cls].append(xb[idx].cpu())
        for k in arms:
            arms[k] = torch.cat(arms[k], 0)
        return arms

    def _query_and_label(self, arms: Dict[int, torch.Tensor], classes: List[int], theta: torch.Tensor, n_samples: int,
                          labeled_dataset: Optional[ConcatDataset], queried_samples: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[ConcatDataset], torch.Tensor]:
        """Queries the oracle for new samples and labels them."""
        counts = self._get_sample_counts(theta, n_samples)
        xs = []
        for i, cls in enumerate(classes):
            if counts[i] > 0 and len(arms[cls]) > 0:
                idx = torch.randint(0, len(arms[cls]), (counts[i],))
                xs.append(arms[cls][idx])
        
        if not xs:
            return None, None, labeled_dataset, queried_samples

        xb = torch.cat(xs, 0).to(self.device)
        yb_oracle = self._query_oracle(xb)

        new_dataset = TensorDataset(xb.cpu(), yb_oracle.cpu())
        if labeled_dataset is None:
            labeled_dataset = new_dataset
        else:
            labeled_dataset = ConcatDataset([labeled_dataset, new_dataset])
        
        queried_samples = torch.cat([queried_samples, xb.cpu()], 0)

        return xb, yb_oracle, labeled_dataset, queried_samples

    def _get_sample_counts(self, theta: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Determines the number of samples to query from each arm based on the policy."""
        if self.policy == 'random':
            p = torch.ones_like(theta) / len(theta)
        else:
            p = torch.softmax(theta / self.temperature, dim=0)
        return torch.multinomial(p, num_samples=n_samples, replacement=True).bincount(minlength=len(theta))

    def _train_student(self, train_loader: DataLoader):
        """Trains the student model for one round."""
        self.student.train()
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss() if self.label_only else self._soft_cross_entropy

        for _ in range(self.epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.student(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

    def _update_theta(self, xb: torch.Tensor, yb_oracle: torch.Tensor, classes: List[int], theta: torch.Tensor, queried_samples: torch.Tensor) -> torch.Tensor:
        """Updates the policy parameters (theta) based on the rewards."""
        rewards = self._calculate_reward(xb, yb_oracle, queried_samples)
        reward_vals = torch.zeros(len(classes))
        counts_per_class = torch.zeros(len(classes))
        y_labels = yb_oracle if self.label_only else yb_oracle.argmax(1)

        for i in range(len(y_labels)):
            try:
                class_idx = classes.index(y_labels[i].item())
                reward_vals[class_idx] += rewards[i]
                counts_per_class[class_idx] += 1
            except ValueError:
                continue
        
        reward_vals /= (counts_per_class + 1e-8)
        reward_vals = (reward_vals - reward_vals.mean()) / (reward_vals.std() + 1e-8)
        return theta + self.policy_lr * reward_vals

    def _calculate_reward(self, xb: torch.Tensor, yb_oracle: torch.Tensor, queried_samples: torch.Tensor) -> torch.Tensor:
        """Calculates rewards for the Knockoff Nets adaptive policy."""
        if self.reward == 'loss':
            if self.label_only:
                return torch.zeros(len(xb))
            with torch.no_grad():
                student_probs = torch.softmax(self.student(xb), 1)
            return -(yb_oracle * torch.log(student_probs + 1e-8)).sum(1)
        elif self.reward == 'cert':
            if self.label_only:
                return torch.zeros(len(xb))
            return yb_oracle.max(1).values

        elif self.reward == 'div':
            if queried_samples.numel() == 0:
                return torch.ones(xb.size(0)) # No previous samples to compare to
            
            # For efficiency, compare against a random subset of previously queried samples
            subset_size = min(self.div_subset_size, queried_samples.size(0))
            subset_indices = torch.randperm(queried_samples.size(0))[:subset_size]
            queried_subset = queried_samples[subset_indices].to(xb.device)

            # Calculate pairwise distances
            dist_matrix = torch.cdist(xb.view(xb.size(0), -1), queried_subset.view(subset_size, -1))
            
            # The reward is the distance to the nearest neighbor in the subset
            min_dists, _ = dist_matrix.min(dim=1)
            return min_dists
        else:
            return torch.zeros(len(xb))

    def _soft_cross_entropy(self, logits, y_soft):
        """Calculates soft cross-entropy loss."""
        return -torch.mean(torch.sum(y_soft * torch.nn.functional.log_softmax(logits, dim=1), dim=1))