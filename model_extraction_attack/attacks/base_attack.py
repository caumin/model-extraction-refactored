
from abc import ABC, abstractmethod
import torch

class BaseAttack(ABC):
    """Abstract base class for all attacks."""

    def __init__(self, student: torch.nn.Module, oracle, query_loader, device):
        self.student = student
        self.oracle = oracle
        self.query_loader = query_loader
        self.device = device
        self.n_queries = 0

    @abstractmethod
    def run(self):
        """Runs the attack."""
        pass

    def _query_oracle(self, x):
        """Queries the oracle and increments the query count."""
        self.n_queries += x.shape[0]
        return self.oracle(x)
