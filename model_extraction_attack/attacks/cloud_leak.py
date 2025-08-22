import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .base_attack import BaseAttack
from typing import Tuple
from ..crafter import fgsm_family_crafter, select_targets
from ..metrics import agreement
from .registry import register_attack

@register_attack("cloud_leak")
class CloudLeakAttack(BaseAttack):
    """Implementation of the CloudLeak model stealing attack.

    This attack focuses on extracting large-scale DNN models from cloud-based platforms.
    """
    def __init__(self, student: nn.Module, oracle, query_loader: DataLoader, device: torch.device,
                 cl_param1: int, cl_param2: float):
        """Initializes the CloudLeakAttack.

        Args:
            student: The student model to be trained.
            oracle: The black-box oracle model.
            query_loader: DataLoader for the initial query dataset.
            device: The device to run the attack on (e.g., 'cpu' or 'cuda').
            cl_param1: Placeholder parameter (e.g., batch size for training).
            cl_param2: Placeholder parameter (e.g., step size for FGSM or learning rate).
        """
        super().__init__(student, oracle, query_loader, device)
        self.cl_param1 = cl_param1 # Number of adversarial examples to generate
        self.cl_param2 = cl_param2 # Step size for FGSM

    def run(self) -> Tuple[nn.Module, int]:
        """Executes the simplified CloudLeak attack.

        Returns:
            A tuple containing the trained student model and the total number of queries made.
        """
        print("Running simplified CloudLeak attack...")

        # Simplified CloudLeak logic:
        # 1. Generate adversarial examples from a subset of query data
        # 2. Query the oracle with adversarial examples
        # 3. Train the student model with adversarial examples and oracle labels

        # Get a batch of data from the query loader
        initial_data_iter = iter(self.query_loader)
        try:
            xb_initial = next(initial_data_iter)[0].to(self.device)
        except StopIteration:
            raise ValueError("Query loader is empty.")

        # Generate adversarial examples using FGSM (simplified)
        # Using cl_param2 as step size for FGSM
        adversarial_examples = fgsm_family_crafter(
            model=self.student, 
            xb=xb_initial, 
            step=self.cl_param2, 
            targeted=False, 
            target_rule="next", 
            steps=1 # FGSM
        )

        # Query the oracle with adversarial examples
        adversarial_labels = self._query_oracle(adversarial_examples)

        # Train the student model with adversarial examples and oracle labels
        adversarial_dataset = TensorDataset(adversarial_examples.cpu(), adversarial_labels.cpu())
        train_loader = DataLoader(adversarial_dataset, batch_size=self.cl_param1, shuffle=True) # Using cl_param1 as batch_size for simplicity

        # Simplified training loop (similar to _train_student in other attacks)
        self.student.train()
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.cl_param2) # Using cl_param2 as lr for simplicity
        loss_fn = nn.CrossEntropyLoss() # Assuming label_only for simplicity

        for epoch in range(1): # Just one epoch for simplicity
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.student(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        # Calculate agreement metrics
        if adversarial_dataset is not None:
            current_train_loader = DataLoader(adversarial_dataset, batch_size=self.cl_param1, shuffle=False) # No shuffle for metrics
            agreement_score = agreement(self.student, self.oracle.model, current_train_loader, self.device)
            print(f"CloudLeak agreement: {agreement_score}")

        return self.student, self.n_queries
