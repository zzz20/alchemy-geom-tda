"""Метрики для дипольного момента, поляризуемости и скалярных свойств."""
import torch
import torch.nn.functional as F
from torch import Tensor


def mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Absolute Error (по всем элементам)."""
    return (pred - target).abs().mean()


def mse(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return mse(pred, target).sqrt()


def dipole_mae(pred_mu: Tensor, target_mu: Tensor) -> Tensor:
    """MAE по каждой компоненте диполя, потом среднее.

    Args:
        pred_mu:   (B, 3)
        target_mu: (B, 3)
    """
    return (pred_mu - target_mu).abs().mean()


def dipole_norm_mae(pred_mu: Tensor, target_mu: Tensor) -> Tensor:
    """MAE на норме диполя |μ|."""
    pred_norm = pred_mu.norm(dim=-1)
    target_norm = target_mu.norm(dim=-1)
    return (pred_norm - target_norm).abs().mean()


def dipole_angular_error(pred_mu: Tensor, target_mu: Tensor, eps: float = 1e-8) -> Tensor:
    """Средняя угловая ошибка в градусах.

    arccos( (μ·μ̂) / (|μ||μ̂|) )
    """
    dot = (pred_mu * target_mu).sum(dim=-1)
    norm_pred = pred_mu.norm(dim=-1) + eps
    norm_target = target_mu.norm(dim=-1) + eps
    cos_sim = (dot / (norm_pred * norm_target)).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    return torch.rad2deg(angle_rad).mean()


def polarizability_iso_mae(pred_alpha: Tensor, target_alpha: Tensor) -> Tensor:
    """MAE на изотропной части tr(α)/3.

    Args:
        pred_alpha:   (B, 3, 3)
        target_alpha: (B, 3, 3)
    """
    pred_iso = pred_alpha.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (B,)
    target_iso = target_alpha.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    return (pred_iso - target_iso).abs().mean()


def polarizability_frobenius_mae(pred_alpha: Tensor, target_alpha: Tensor) -> Tensor:
    """MAE по Фробениусовой норме разности тензоров."""
    return (pred_alpha - target_alpha).abs().mean()


def polarizability_aniso_mae(pred_alpha: Tensor, target_alpha: Tensor) -> Tensor:
    """MAE на анизотропной (симметричной бесследовой) части α_aniso = α - tr(α)/3 · I."""
    pred_iso = pred_alpha.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (B,1)
    target_iso = target_alpha.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)

    I = torch.eye(3, dtype=pred_alpha.dtype, device=pred_alpha.device)  # (3,3)
    pred_aniso = pred_alpha - pred_iso[..., None] * I
    target_aniso = target_alpha - target_iso[..., None] * I
    return (pred_aniso - target_aniso).abs().mean()


METRIC_REGISTRY = {
    "mae": mae,
    "rmse": rmse,
    "dipole_mae": dipole_mae,
    "dipole_norm_mae": dipole_norm_mae,
    "dipole_angular_error": dipole_angular_error,
    "alpha_iso_mae": polarizability_iso_mae,
    "alpha_frob_mae": polarizability_frobenius_mae,
    "alpha_aniso_mae": polarizability_aniso_mae,
}
