from typing import Dict, Any, Tuple
import numpy as np, torch
from sklearn.metrics import f1_score

def calculate_papernot_transferability(y_v_new: torch.Tensor, y_s_exist: torch.Tensor, y_s_new: torch.Tensor, y_tgt: torch.Tensor) -> Dict[str, float]:
    """
    Calculates transferability metrics for Papernot-style attacks.
    Args:
        y_v_new: Oracle's labels for the newly generated samples.
        y_s_exist: Student's predictions for the existing samples (before augmentation).
        y_s_new: Student's predictions for the newly generated samples.
        y_tgt: Target labels for the newly generated samples.
    Returns:
        A dictionary containing 'transferability_untargeted', 'transferability_targeted', and 'rnd_agreement'.
    """
    # NOTE: 'transferability_disagreement' is implemented here. The PRADA paper's definition of
    # non-targeted transferability is about the adversarial example causing the *target model*
    # (oracle) to misclassify from its original class. This metric measures the proportion of
    # samples where the oracle's new label is *different* from the student's old prediction.
    disagreement = (y_v_new != y_s_exist.cpu()).float().mean().item()
    # Targeted transferability: Oracle's new label == Target label
    targeted_ok = (y_v_new == y_tgt.cpu()).float().mean().item()
    # RND-agreement: Student's new prediction == Oracle's new label
    rnd_agree = (y_s_new == y_v_new).float().mean().item()

    return {
        "transferability_untargeted": float(disagreement),
        "transferability_targeted": float(targeted_ok),
        "rnd_agreement": float(rnd_agree),
    }

@torch.no_grad()
def agreement(model_a, model_b, data_loader, device="cpu") -> float:
    """
    Calculates the agreement between two models' predictions.
    """
    model_a.eval()
    model_b.eval()
    agreements = []
    for xb, _ in data_loader:
        xb = xb.to(device)
        logits_a = model_a(xb)
        logits_b = model_b(xb)
        preds_a = torch.argmax(logits_a, dim=1)
        preds_b = torch.argmax(logits_b, dim=1)
        agreements.append((preds_a == preds_b).float().cpu())
    return torch.cat(agreements).mean().item()

@torch.no_grad()
def test_agreement(student_model: torch.nn.Module, oracle_model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: str) -> Dict[str, float]:
    """
    Calculates the test agreement (macro-averaged F1-score) between student and oracle models.
    """
    student_model.eval()
    oracle_model.eval()
    
    all_oracle_preds = []
    all_student_preds = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        oracle_preds = torch.argmax(oracle_model(xb), dim=1).cpu().numpy()
        student_preds = torch.argmax(student_model(xb), dim=1).cpu().numpy()
        
        all_oracle_preds.extend(oracle_preds)
        all_student_preds.extend(student_preds)

    f1_macro = f1_score(all_oracle_preds, all_student_preds, average='macro')
    return {"test_agreement_f1_macro": float(f1_macro)}

@torch.no_grad()
def ru_agreement(student_model: torch.nn.Module, oracle_model: torch.nn.Module, num_samples: int, img_shape: Tuple[int, ...], device: str) -> Dict[str, float]:
    """
    Calculates the Random Uniform (RU) agreement between student and oracle models.
    Generates uniformly random samples in the input space [0, 1].
    """
    student_model.eval()
    oracle_model.eval()

    # Generate random uniform samples
    random_samples = torch.rand(num_samples, *img_shape).to(device)

    oracle_preds = torch.argmax(oracle_model(random_samples), dim=1).cpu().numpy()
    student_preds = torch.argmax(student_model(random_samples), dim=1).cpu().numpy()

    # Calculate accuracy (simple agreement on random samples)
    accuracy = np.mean(oracle_preds == student_preds)
    return {"ru_agreement_accuracy": float(accuracy)}