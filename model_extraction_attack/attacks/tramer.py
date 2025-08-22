import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .base_attack import BaseAttack
from typing import Literal, Optional, Tuple
import math
from collections import defaultdict, deque
import random
from ..metrics import agreement
from .registry import register_attack
from pathlib import Path # Added import for Path

@register_attack("tramer")
class TramerAttack(BaseAttack):
    """Implementation of the Tramèr et al. model extraction attack.

    This attack trains a substitute model by adaptively querying the target oracle
    using different strategies (uniform, adaptive, linesearch).
    """
    def __init__(self, student: nn.Module, oracle, query_loader: DataLoader, device: torch.device,
                 rounds: int, queries_per_round: int, epochs: int, batch_size: int, lr: float,
                 label_only: bool, strategy: Literal["uniform", "linesearch", "adaptive"], candidate_factor: float,
                 ls_steps: int, ls_jitter: float = 0.0, tr_seed_ratio: float = 0.25, tr_max_passes: int = 3,
                 run_dir: Optional[Path] = None):
        """Initializes the TramerAttack.

        Args:
            student: The student model to be trained.
            oracle: The black-box oracle model.
            query_loader: DataLoader for the initial query dataset.
            device: The device to run the attack on (e.g., 'cpu' or 'cuda').
            rounds: Number of rounds for the attack.
            queries_per_round: Number of queries to make in each round.
            epochs: Number of epochs to train the student in each round.
            batch_size: Batch size for student training.
            lr: Learning rate for student training.
            label_only: If True, oracle provides only labels; otherwise, probabilities.
            strategy: Query strategy ('uniform', 'linesearch', or 'adaptive').
            candidate_factor: Factor for candidate generation in adaptive strategy.
            ls_steps: Number of steps for binary search in linesearch strategy.
            ls_jitter: Jitter for boundary points in linesearch strategy.
            tr_seed_ratio: Ratio of initial seed samples for linesearch strategy.
            tr_max_passes: Maximum passes for collecting opposite pairs in linesearch strategy.
            run_dir: Optional run directory for saving attack-specific outputs.
        """
        super().__init__(student, oracle, query_loader, device)
        self.rounds = rounds
        self.queries_per_round = queries_per_round
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = float(lr)
        self.label_only = label_only
        self.strategy = strategy
        self.candidate_factor = candidate_factor
        self.ls_steps = ls_steps
        self.ls_jitter = ls_jitter
        self.tr_seed_ratio = tr_seed_ratio
        self.tr_max_passes = tr_max_passes
        self.metrics_history = [] # Initialize for logging metrics
        self.run_dir = run_dir # Store run_dir

    def run(self) -> Tuple[nn.Module, int]:
        """Executes the Tramèr model extraction attack.

        Returns:
            A tuple containing the trained student model and the total number of queries made.
        """
        labeled_dataset = None
        for r in range(self.rounds):
            need = self.queries_per_round
            if self.strategy == 'uniform':
                xb, yb = self._uniform_strategy()
            elif self.strategy == 'adaptive':
                xb, yb = self._adaptive_strategy()
            elif self.strategy == 'linesearch':
                xb, yb = self._linesearch_strategy(need)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            new_dataset = TensorDataset(xb.cpu(), yb.cpu())
            if labeled_dataset is None:
                labeled_dataset = new_dataset
            else:
                labeled_dataset = ConcatDataset([labeled_dataset, new_dataset])

            train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True)
            self._train_student(train_loader)

            # Calculate agreement metrics
            if labeled_dataset is not None:
                current_train_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=False) # No shuffle for metrics
                agreement_score = agreement(self.student, self.oracle.model, current_train_loader, self.device)
                
                # Log metrics
                round_metrics = {
                    "round": r,
                    "n_queries": self.n_queries,
                    "agreement_score": agreement_score,
                    "current_labeled_dataset_size": len(labeled_dataset),
                }
                self.metrics_history.append(round_metrics)
                print(f"Round {r} metrics: {round_metrics}")

        # Save metrics history to a JSON file
        if self.metrics_history:
            import json
            from pathlib import Path
            output_path = self.run_dir / "tramer_metrics.json" # Use run_dir
            with open(output_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            print(f"Metrics history saved to {output_path}")

        return self.student, self.n_queries

    def _uniform_strategy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implements the uniform query strategy."""
        xb = self._gather_uniform(self.queries_per_round)
        yb = self._query_oracle(xb)
        return xb, yb

    def _adaptive_strategy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implements the adaptive query strategy."""
        xb = self._gather_adaptive(self.queries_per_round)
        yb = self._query_oracle(xb)
        return xb, yb

    def _linesearch_strategy(self, need: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implements the linesearch query strategy."""
        X_seed = None; y_seed = None; n_seed = 0
        if self.tr_seed_ratio and self.tr_seed_ratio > 0.0:
            n_seed = max(1, int(need * float(self.tr_seed_ratio)))
            X_seed = self._gather_uniform(n_seed)
            y_seed = self._query_oracle(X_seed)
            self.n_queries += int(X_seed.size(0))

        need_ls = max(0, need - n_seed)
        if need_ls > 0:
            X_ls, y_ls, used = self._gather_linesearch(
                need_ls,
                steps=self.ls_steps, jitter=self.ls_jitter,
                max_passes=self.tr_max_passes
            )
            self.n_queries += int(used)
        else:
            X_ls = None; y_ls = None

        parts_x, parts_y = [], []
        if X_seed is not None:
            parts_x.append(X_seed.detach().cpu())
            parts_y.append((y_seed.long() if self.label_only else y_seed.float()).cpu())
        if X_ls is not None:
            parts_x.append(X_ls.detach().cpu())
            parts_y.append((y_ls.long() if self.label_only else y_ls.float()).cpu())
        
        if not parts_x:
            raise RuntimeError("Linesearch strategy yielded no samples.")

        X = torch.cat(parts_x, 0)
        y = torch.cat(parts_y, 0)
        return X, y

    @torch.no_grad()
    def _iter_unlabeled_batches(self):
        """Yields batches of unlabeled data from the query loader."""
        for batch in self.query_loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            yield xb.to(self.device, non_blocking=True)

    def _gather_uniform(self, n_samples: int) -> torch.Tensor:
        """Gathers a specified number of samples uniformly from the query pool."""
        xs = []
        tot = 0
        for xb in self._iter_unlabeled_batches():
            xs.append(xb)
            tot += xb.size(0)
            if tot >= n_samples:
                break
        if not xs:
            raise ValueError("Pool loader yielded no batches.")
        return torch.cat(xs, 0)[:n_samples].to(self.device)

    def _gather_adaptive(self, n_samples: int) -> torch.Tensor:
        """Gathers a specified number of samples adaptively based on student uncertainty."""
        candidates = self._gather_uniform(int(n_samples * self.candidate_factor))
        scores = self._score_uncertainty(candidates)
        _, indices = torch.topk(scores, min(n_samples, candidates.size(0)), largest=True)
        return candidates[indices]

    def _score_uncertainty(self, xb: torch.Tensor) -> torch.Tensor:
        """Calculates the uncertainty score for a batch of samples."""
        self.student.eval()
        with torch.no_grad():
            logits = self.student(xb)
            probs = torch.softmax(logits, 1)
            return 1 - probs.max(1).values

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

    def _soft_cross_entropy(self, logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
        """Calculates soft cross-entropy loss."""
        return -torch.mean(torch.sum(y_soft * torch.nn.functional.log_softmax(logits, dim=1), dim=1))

    @torch.no_grad()
    def _collect_opposite_pairs(self, want_pairs: int,
                                mode: Literal["label","probs"], max_passes: int = 3) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Builds label-opposite pairs (A,B) in a class-balanced, randomized way for linesearch.
        Returns (A, B, queries_used).
        """
        buckets: dict[int, deque] = defaultdict(deque)  # label -> queue of samples (on device)
        used = 0
        A_list, B_list = [], []

        def _labels(x: torch.Tensor) -> torch.Tensor:
            y = self._query_oracle(x)   # (B,) for label-only, or (B,C) for probs → convert to labels on the SAME device
            return (y if mode == "label" else y.argmax(1)).to(self.device)

        for _ in range(max_passes):
            for xb in self._iter_unlabeled_batches():
                yb = _labels(xb); used += xb.size(0)
                # push to per-class queues
                for lbl in yb.unique():
                    lbl_i = int(lbl.item())
                    idx = (yb == lbl).nonzero(as_tuple=False).squeeze(1)
                    # 분할 push (detach는 불필요, device 유지)
                    for t in xb[idx]:
                        buckets[lbl_i].append(t.unsqueeze(0))  # (1,C,H,W)

                # 가능해질 때마다 균형 있게 pair 뽑기
                labels = [k for k,v in buckets.items() if len(v) > 0]
                while (len(A_list) < want_pairs) and (len(labels) >= 2):
                    l1, l2 = random.sample(labels, 2)
                    a = buckets[l1].popleft()  # (1,C,H,W)
                    b = buckets[l2].popleft()
                    A_list.append(a)
                    B_list.append(b)
                    labels = [k for k,v in buckets.items() if len(v) > 0]
                if len(A_list) >= want_pairs:
                    break
            if len(A_list) >= want_pairs:
                break

        if len(A_list) == 0:
            raise RuntimeError("Could not form opposite-label pairs; pool too small or labels too imbalanced.")

        A = torch.cat(A_list, 0).to(self.device, non_blocking=True)
        B = torch.cat(B_list, 0).to(self.device, non_blocking=True)
        return A, B, used

    @torch.no_grad()
    def _bisect_boundary(self, xa: torch.Tensor, xb: torch.Tensor, steps: int,
                         mode: Literal["label","probs"]) -> Tuple[torch.Tensor, int]:
        """
        Performs binary line-search between two points until the label flips.
        Returns the near-boundary point(s) and the number of oracle queries used.
        """
        dev = self.device
        used = 0

        # Ensure a/b are on the same device and have grad OFF
        a = xa.to(dev).detach()
        b = xb.to(dev).detach()

        def lab(t: torch.Tensor) -> torch.Tensor:
            y = self._query_oracle(t)   # (B,) for label-only, or (B,C) for probs → convert to labels on the SAME device
            return (y if mode == "label" else y.argmax(1)).to(dev)

        # initial labels
        ya_lab = lab(a); used += a.size(0)
        yb_lab = lab(b); used += b.size(0)

        for _ in range(int(steps)):
            mid = (a + b) * 0.5                        # (B, C, H, W) on dev
            ymid_lab = lab(mid); used += mid.size(0)   # (B,) on dev

            same_as_a = (ymid_lab == ya_lab)           # (B,) on dev (bool)
            # expand mask to image shape
            cond = same_as_a.view(-1, *([1] * (a.dim() - 1)))  # (B,1,1,1) on dev, bool
            ncond = ~cond

            # Move the endpoint that has the SAME label as mid to mid
            a = torch.where(cond, mid, a)
            ya_lab = torch.where(same_as_a, ymid_lab, ya_lab)

            b = torch.where(ncond, mid, b)
            yb_lab = torch.where(~same_as_a, ymid_lab, yb_lab)

        # final midpoint as near-boundary sample
        mid = (a + b) * 0.5
        return mid.detach(), used

    def _gather_linesearch(self, need: int,
                           steps: int = 5, mode: Literal["label","probs"] = "label",
                           jitter: float = 0.0, max_passes: int = 3) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generates samples near the decision boundary using the linesearch strategy.
        Returns (X_new, Y_oracle, queries_used).
        """
        # 한 쌍당 1개의 near-boundary 샘플을 목표로 함
        want_pairs = max(1, math.ceil(need / 1))

        # 1) 서로 다른 라벨의 시드 쌍 수집 (A,B는 pool_loader에서 나온 device로)
        A, B, used_pairs = self._collect_opposite_pairs(want_pairs, mode, max_passes=max_passes)

        # 2) 각 쌍에 대해 이분 탐색
        mids, used_bis = [], 0
        for i in range(A.size(0)):
            # A/B를 명시적으로 device에 맞춤 (혹시 모를 cpu 텐서 방지)
            ai = A[i:i+1].to(self.device, non_blocking=True)
            bi = B[i:i+1].to(self.device, non_blocking=True)

            mid, u = self._bisect_boundary(ai, bi, steps=steps, mode=mode)
            used_bis += u

            if jitter > 0.0:
                noise = torch.randn_like(mid) * jitter    # same device
                mid = (mid + noise).clamp(0, 1)

            mids.append(mid.detach())
            if len(mids) >= need:
                break

        X_mid = torch.cat(mids, 0)                       # on device
        # 3) near-boundary 점들에 대해 라벨링
        y_mid = self._query_oracle(X_mid)  # on device
        used_bis += X_mid.size(0)

        return X_mid.detach(), y_mid.detach().cpu(), (used_pairs + used_bis)