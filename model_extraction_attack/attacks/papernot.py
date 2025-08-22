from typing import Literal, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .base_attack import BaseAttack
from ..metrics import calculate_papernot_transferability
from ..crafter import select_targets
from .registry import register_attack
from ..utils import soft_cross_entropy # Import centralized soft_cross_entropy

@register_attack("papernot")
class PapernotAttack(BaseAttack):
    """Implementation of the Papernot et al. model extraction attack.

    This attack trains a substitute model by querying the target oracle and generating
    synthetic data points near the decision boundary.
    """
    def __init__(self, student: nn.Module, oracle, query_loader: DataLoader, device: torch.device,
                 rounds: int, lambda_aug: float, epochs: int, batch_size: int, lr: float,
                 label_only: bool, reservoir: Optional[int], crafter: str, target_policy: Literal["next","least","random"]):
        """Initializes the PapernotAttack.

        Args:
            student: The student model to be trained.
            oracle: The black-box oracle model.
            query_loader: DataLoader for the initial query dataset.
            device: The device to run the attack on (e.g., 'cpu' or 'cuda').
            rounds: Number of rounds for the attack.
            lambda_aug: Step size for adversarial example generation.
            epochs: Number of epochs to train the student in each round.
            batch_size: Batch size for student training.
            lr: Learning rate for student training.
            label_only: If True, oracle provides only labels; otherwise, probabilities.
            reservoir: Maximum number of samples to keep in the reservoir.
            crafter: Type of adversarial example generation method (e.g., 'jbda').
            target_policy: Policy for selecting target labels for adversarial examples.
        """
        super().__init__(student, oracle, query_loader, device)
        self.rounds = rounds
        self.lambda_aug = lambda_aug
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = float(lr)
        self.label_only = label_only
        self.reservoir = reservoir
        self.crafter = crafter
        self.target_policy = target_policy

    def run(self) -> Tuple[nn.Module, int]:
        """Executes the Papernot model extraction attack.

        Returns:
            A tuple containing the trained student model and the total number of queries made.
        """
        # 0. Initial data labeling
        xs, ys = [], []
        for batch in self.query_loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device)
            y = self._query_oracle(xb)
            xs.append(xb.cpu())
            ys.append(y.cpu())
        
        x_pool = torch.cat(xs, 0)
        y_pool = torch.cat(ys, 0)
        labeled_dataset = TensorDataset(x_pool, y_pool)

        for r in range(self.rounds):
            # 1. Train student model
            train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True)
            self._train_student(train_loader)

            # 2. Create new samples
            x_new = self._create_new_samples(x_pool)

            # 3. Query oracle for new samples
            y_new = self._query_oracle(x_new.to(self.device))

            # 4. Add new samples to the dataset
            new_dataset = TensorDataset(x_new.cpu(), y_new.cpu())
            labeled_dataset = ConcatDataset([labeled_dataset, new_dataset])
            x_pool = torch.cat([x_pool, x_new.cpu()], 0)

            # Calculate metrics
            y_v_new = y_new.long().view(-1).cpu() if self.label_only else y_new.softmax(1).argmax(1).cpu()
            with torch.no_grad():
                y_s_new = self.student(x_new.to(self.device)).softmax(1).argmax(1).cpu()
                
                # y_s_exist and y_tgt should be based on the samples that were used to generate x_new
                # In our simplified implementation, x_new is generated from x_pool (which is the current pool).
                # So, y_s_exist and y_tgt should be based on the same samples as x_new.
                y_s_exist = self.student(x_new.to(self.device)).argmax(1).cpu()
                y_tgt = select_targets(self.student(x_new.to(self.device)), mode=self.target_policy).cpu()

            metrics = calculate_papernot_transferability(y_v_new=y_v_new, y_s_exist=y_s_exist, y_s_new=y_s_new, y_tgt=y_tgt)
            print(f"Round {r} metrics: {metrics}")

        return self.student, self.n_queries

    def _train_student(self, train_loader: DataLoader):
        """Trains the student model for one round."""
        self.student.train()
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss() if self.label_only else soft_cross_entropy # Use imported soft_cross_entropy

        for _ in range(self.epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.student(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

    def _create_new_samples(self, x_pool: torch.Tensor) -> torch.Tensor:
        """Creates new samples for the next round by adversarial perturbation."""
        if self.reservoir and len(x_pool) > self.reservoir:
            perm = torch.randperm(len(x_pool))
            x_pool = x_pool[perm[:self.reservoir]]

        x_new = self._crafter_batch(x_pool.to(self.device))
        return x_new

    def _crafter_batch(self, xb: torch.Tensor) -> torch.Tensor:
        """Applies the selected adversarial example crafting method to a batch."""
        if self.crafter == 'jbda':
            return self._jbda_batch(xb)
        else:
            raise NotImplementedError(f"Crafter {self.crafter} not implemented for PapernotAttack.")

    def _jbda_batch(self, xb: torch.Tensor) -> torch.Tensor:
        """Generates adversarial examples using the Jacobian-based Data Augmentation (JBDA) method."""
        x_adv = xb.clone().detach()
        self.student.eval()
        with torch.no_grad():
            logits = self.student(xb)
            y_target = select_targets(self.student(xb), mode=self.target_policy)

        for i in range(len(xb)):
            x_sample = xb[i:i+1].clone().detach().requires_grad_(True)
            for _ in range(int(1/self.lambda_aug)):
                if self.student(x_sample).argmax(1) == y_target[i]:
                    break
                
                logits = self.student(x_sample)
                loss = nn.CrossEntropyLoss()(logits, y_target[i:i+1])
                loss.backward()
                
                grad = x_sample.grad.data.sign()
                x_sample.data = x_sample.data + self.lambda_aug * grad
                x_sample.data = torch.clamp(x_sample.data, 0, 1)
                x_sample.grad.data.zero_()
            x_adv[i] = x_sample.detach()
        return x_adv