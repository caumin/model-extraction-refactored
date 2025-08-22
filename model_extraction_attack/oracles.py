"""Module for defining the Oracle class, which simulates a black-box model API."""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn

class Oracle(nn.Module):
    """Simulates a black-box model API, providing predictions based on specified disclosure modes."""
    def __init__(self, model: nn.Module, disclosure: str = "label",
                 round_decimals: Optional[int] = None, topk: int = 1):
        """Initializes the Oracle.

        Args:
            model: The underlying PyTorch model to query.
            disclosure: The type of information disclosed by the oracle ('label', 'probs', 'logits', 'topk').
            round_decimals: Number of decimal places to round probabilities to (if applicable).
            topk: Number of top predictions to return for 'topk' disclosure mode.
        """
        super().__init__()
        self.model = model.eval()
        self.disclosure = disclosure
        self.round_decimals = round_decimals
        self.topk = topk

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """Queries the oracle with input data.

        Args:
            x: Input tensor.
            mode: Override the default disclosure mode for this query.

        Returns:
            The oracle's response based on the disclosure mode.

        Raises:
            ValueError: If an unknown disclosure mode is specified.
        """
        mode = mode or self.disclosure
        logits = self.model(x)
        if mode == "logits":
            return logits
        probs = torch.softmax(logits, 1)
        if self.round_decimals is not None:
            probs = torch.round(probs * (10**self.round_decimals)) / (10**self.round_decimals)
        if mode in ("probs", "prob", "posterior"):
            return probs
        top1 = probs.argmax(1)
        if mode in ("label","top1"):
            return top1
        if mode == "topk":
            v, i = probs.topk(k=min(self.topk, probs.size(1)), dim=1)
            return i
        raise ValueError(f"Unknown disclosure mode: {mode}")