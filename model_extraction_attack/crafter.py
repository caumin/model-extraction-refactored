"""Module for generating various types of synthetic samples (crafters)."""

import torch
from torch import nn
from typing import Literal, Optional

@torch.no_grad()
def select_targets(logits: torch.Tensor,
                    mode: Literal["next","least","random"]="next") -> torch.Tensor:
    """Selects target labels for adversarial example generation based on the specified mode."""
    probs = torch.softmax(logits, 1)
    y_pred = probs.argmax(1)
    C = logits.shape[1]
    if C == 2:
        return 1 - y_pred
    if mode == "least":
        return probs.argmin(1)
    if mode == "random":
        B = logits.size(0)
        r = torch.randint(low=0, high=C-1, size=(B,), device=logits.device)
        return torch.where(r < y_pred, r, r + 1)   # pick != y_pred
    return (y_pred + 1) % C  # "next"

def fgsm_family_crafter(
    model: nn.Module,
    xb: torch.Tensor,
    *,
    step: float,                 # 스텝 크기(alpha)
    steps: int = 1,              # 반복 횟수 (1 == FGSM)
    momentum: float = 0.0,       # >0면 MI-FGSM
    targeted: bool = False,      # True: 타깃으로, False: 언타깃(멀어지기)
    target_rule: Literal["next","least","random"]="next",
    eps: Optional[float] = None, # L_inf 예산(선택)
    rand_start: bool = False,    # True면 [-eps,eps] 랜덤 스타트 (eps 필요)
) -> torch.Tensor:
    """
    Implements common logic for FGSM, I-FGSM, and MI-FGSM adversarial example generation.
    Inputs and outputs are clamped to [0,1] range.
    """
    model.eval()
    dev = next(model.parameters()).device
    x0  = xb.to(dev)
    x   = x0.clone()

    if rand_start and eps is not None and eps > 0:
        x = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)

    mu = float(max(0.0, momentum))
    g  = torch.zeros_like(x)

    # 타깃 고정 (targeted인 경우)
    if targeted:
        with torch.no_grad():
            y_tgt = select_targets(model(x0), mode=target_rule)

    for _ in range(max(1, int(steps))):
        x_req = x.detach().requires_grad_(True)
        logits = model(x_req)

        if targeted:
            loss = nn.CrossEntropyLoss()(logits, y_tgt)
            dir_sign = -1.0  # 타깃으로 가려면 -grad 방향
        else:
            y_cur = logits.argmax(1)
            loss  = nn.CrossEntropyLoss()(logits, y_cur)
            dir_sign = +1.0  # 현재 클래스에서 멀어지기

        loss.backward()
        grad = x_req.grad.detach()

        if mu > 0:
            # MI-FGSM: 정규화 + 모멘텀 누적
            grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)
            g = mu * g + grad
            delta = dir_sign * step * g.sign()
        else:
            delta = dir_sign * step * grad.sign()

        x = x + delta

        # L_inf 프로젝션(선택)
        if eps is not None and eps > 0:
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)

        x = x.clamp(0, 1)

        # 조기 종료(타깃 달성)
        if targeted:
            with torch.no_grad():
                if model(x).argmax(1).eq(y_tgt).all():
                    break

    return x.detach()

def color_aug_batch(xb: torch.Tensor, lam: float) -> torch.Tensor:
    """Applies a simple color/domain augmentation to a batch."""
    dev = xb.device
    x_out = xb.clone().detach()
    for i in range(xb.size(0)):
        x_sample = xb[i]
        if x_sample.size(0) == 1:  # grayscale
            perturbation = (torch.rand(1, device=dev) - 0.5) * 2 * lam
            x_out[i] = (x_sample + perturbation).clamp(0, 1)
        else:  # color
            ch = torch.randint(0, x_sample.size(0), (1,), device=dev).item()
            perturbation = (torch.rand(1, device=dev) - 0.5) * 2 * lam
            x_out[i, ch] = (x_sample[ch] + perturbation).clamp(0, 1)
    return x_out

def jsma_batch(model: nn.Module, xb: torch.Tensor, k: int, lam: float, target_policy: Literal["next","least","random"]) -> torch.Tensor:
    """Generates adversarial examples using the Jacobian Saliency Map Attack (JSMA) method."""
    model.eval()
    B = xb.size(0)
    x_out = xb.clone().detach()

    # 타깃 라벨 벡터(배치) 미리 계산
    with torch.no_grad():
        y_tgt_all = select_targets(model(xb), mode=target_policy)

    for i in range(B):
        x_adv = xb[i:i+1].clone().detach()
        y_tgt_scalar = int(y_tgt_all[i].item())

        x_req = x_adv.detach().requires_grad_(True)
        logits = model(x_req)
        # 이미 타깃이라면 skip
        if int(logits.argmax(1)[0].item()) != y_tgt_scalar:
            target_logit = logits[0, y_tgt_scalar]
            target_logit.backward()
            g = x_req.grad.detach()
            gabs = g.abs().view(-1)
            k_use = min(k, gabs.numel())
            topk = torch.topk(gabs, k=k_use, largest=True).indices
            mask = torch.zeros_like(gabs); mask[topk] = 1.0
            mask = mask.view_as(g)
            step = lam * g.sign() * mask
            x_adv = (x_adv + step).clamp(0, 1)

        x_out[i] = x_adv
    return x_out.detach()