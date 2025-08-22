import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .base_attack import BaseAttack
from typing import Literal, Tuple, Optional
from ..metrics import calculate_papernot_transferability, test_agreement, ru_agreement
from ..crafter import select_targets, fgsm_family_crafter, color_aug_batch, jsma_batch
from ..data import make_seed_from_labeled
from .registry import register_attack
from pathlib import Path # Added import for Path

@register_attack("prada")
class PradaAttack(BaseAttack):
    """Implementation of the PRADA model stealing attack.

    This attack orchestrates different model extraction techniques within the PRADA framework.
    It focuses on generating effective synthetic queries and optimizing training hyperparameters.
    """
    def __init__(self, student: nn.Module, oracle, query_loader: DataLoader, device: torch.device,
                 prada_config: dict,
                 query_budget: Optional[int] = None,
                 optimizer_type: str = "adam",
                 optimizer_params: Optional[dict] = None,
                 test_loader: Optional[DataLoader] = None,
                 img_shape: Optional[Tuple[int, ...]] = None,
                 run_dir: Optional[Path] = None,
                 cv_search_enabled: bool = False): # New parameter for CV-SEARCH
        super().__init__(student, oracle, query_loader, device)
        self.prada_config = prada_config

        # Extract common parameters from prada_config
        self.rounds = prada_config.get("rounds", 1)
        self.lambda_aug = prada_config.get("lambda_aug", 0.1)
        self.epochs = prada_config.get("epochs", 1)
        self.batch_size = prada_config.get("batch_size", 1)
        self.lr = float(prada_config.get("lr", 1e-3))
        self.label_only = prada_config.get("label_only", True)
        self.reservoir = prada_config.get("reservoir", None) # Set to None by default to disable capping
        self.crafter = prada_config.get("crafter", "jbda")
        self.target_policy = prada_config.get("target_policy", "next")
        self.jsma_k = prada_config.get("jsma_k", 32)
        self.pn_targeted = prada_config.get("pn_targeted", False)
        self.pn_steps = prada_config.get("pn_steps", 1)
        self.pn_momentum = prada_config.get("pn_momentum", 0.0)
        self.pn_eps = prada_config.get("pn_eps", None)
        self.pn_rand_start = prada_config.get("pn_rand_start", False)
        self.seed_per_class = prada_config.get("seed_per_class", None) # New parameter for initial seed size

        self.query_budget = query_budget
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.test_loader = test_loader
        self.img_shape = img_shape
        self.run_dir = run_dir # Store run_dir
        self.cv_search_enabled = cv_search_enabled # Store CV-SEARCH flag

    def run(self) -> Tuple[nn.Module, int]:
        print("Running PRADA attack...")

        self.metrics_history = [] # Initialize for logging metrics

        # 0. Initial data labeling with seed samples
        if self.seed_per_class is not None:
            # Create a seed DataLoader from the query_loader (full dataset)
            seed_loader = make_seed_from_labeled(self.query_loader, self.seed_per_class, seed=42) # Use a fixed seed for reproducibility
            xs, ys = [], []
            for batch in seed_loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(self.device)
                y = self._query_oracle(xb)
                xs.append(xb.cpu())
                ys.append(y.cpu())
            x_pool = torch.cat(xs, 0)
            y_pool = torch.cat(ys, 0)
            labeled_dataset = TensorDataset(x_pool, y_pool)
        else:
            # Fallback to using the entire query_loader for initial labeling (current behavior)
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
            # Check query budget before generating new samples
            if self.query_budget is not None and self.n_queries >= self.query_budget:
                print(f"Query budget {self.query_budget} reached. Stopping attack.")
                break

            # 1. Train student model
            train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True)
            self._train_student(train_loader)

            # 2. Create new samples using PRADA's synthetic query generation
            # If reservoir is None, x_pool is not capped, allowing doubling
            current_x_pool_for_crafter = x_pool
            if self.reservoir is not None and len(x_pool) > self.reservoir:
                perm = torch.randperm(len(x_pool))
                current_x_pool_for_crafter = x_pool[perm[:self.reservoir]]

            x_new = self._create_new_samples(current_x_pool_for_crafter)

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
                y_s_exist = self.student(x_new.to(self.device)).argmax(1).cpu()
                y_tgt = select_targets(self.student(x_new.to(self.device)), mode=self.target_policy).cpu()

            metrics = calculate_papernot_transferability(y_v_new=y_v_new, y_s_exist=y_s_exist, y_s_new=y_s_new, y_tgt=y_tgt)
            
            # Calculate additional PRADA metrics
            test_agree_metrics = {}
            if self.test_loader is not None:
                test_agree_metrics = test_agreement(self.student, self.oracle.model, self.test_loader, self.device)
            
            ru_agree_metrics = {}
            if self.img_shape is not None:
                ru_agree_metrics = ru_agreement(self.student, self.oracle.model, num_samples=4000, img_shape=self.img_shape, device=self.device)

            # Log metrics
            round_metrics = {
                "round": r,
                "n_queries": self.n_queries,
                "papernot_transferability": metrics,
                "current_x_pool_size": len(x_pool), # Log current x_pool size
                **test_agree_metrics,
                **ru_agree_metrics,
            }
            self.metrics_history.append(round_metrics)
            print(f"Round {r} metrics: {round_metrics}")

        # Save metrics history to a JSON file
        if self.metrics_history:
            import json
            from pathlib import Path
            output_path = self.run_dir / "prada_metrics.json" # Use run_dir
            with open(output_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            print(f"Metrics history saved to {output_path}")

        return self.student, self.n_queries

    def _train_student(self, train_loader: DataLoader):
        """Trains the student model for one round."""
        self.student.train()
        
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, **self.optimizer_params)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.student.parameters(), lr=self.lr, **self.optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        loss_fn = nn.CrossEntropyLoss() if self.label_only else self._soft_cross_entropy

        for _ in range(self.epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.student(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

    def _create_new_samples(self, x_pool: torch.Tensor) -> torch.Tensor:
        """Creates new samples for the next round using PRADA's synthetic query generation approach."""
        # If reservoir is None, x_pool is not capped, allowing doubling
        current_x_pool_for_crafter = x_pool
        if self.reservoir is not None and len(x_pool) > self.reservoir:
            perm = torch.randperm(len(x_pool))
            current_x_pool_for_crafter = x_pool[perm[:self.reservoir]]

        if self.crafter == 'jbda':
            return self._jbda_batch(current_x_pool_for_crafter.to(self.device))
        elif self.crafter in ("fgsm","ifgsm","mifgsm","fgsm_u","fgsm_t","mi-fgsm_u","mi-fgsm_t","n-ifgsm","ti-fgsm"):
            return fgsm_family_crafter(self.student, current_x_pool_for_crafter.to(self.device), step=self.lambda_aug, targeted=self.pn_targeted, target_rule=self.target_policy, steps=self.pn_steps, momentum=self.pn_momentum, eps=self.pn_eps, rand_start=self.pn_rand_start)
        elif self.crafter == "color":
            return color_aug_batch(current_x_pool_for_crafter.to(self.device), self.lambda_aug)
        elif self.crafter == "jsma":
            return jsma_batch(self.student, current_x_pool_for_crafter.to(self.device), self.jsma_k, self.lambda_aug, self.target_policy)
        else:
            raise NotImplementedError(f"Crafter {self.crafter} not implemented for PradaAttack.")

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

    def _soft_cross_entropy(self, logits, y_soft):
        """Calculates soft cross-entropy loss."""
        return -torch.mean(torch.sum(y_soft * nn.functional.log_softmax(logits, dim=1), dim=1))