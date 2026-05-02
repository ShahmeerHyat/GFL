"""
losses.py
=========

The Generalized Focal Loss (GFL) family and baseline losses.

References
----------
- Cross-Entropy:        CE(p_t)   = -log(p_t)
- Focal Loss [Lin+17]:  FL(p_t)   = -(1 - p_t)^gamma * log(p_t)
- GCE [Zhang+18]:       GCE(p_t)  = (1 - p_t^q) / q
- MAE [Ghosh+17]:       MAE(p_t)  = 1 - p_t   (multi-class form)
- GFL (this paper):     GFL(p_t)  = -phi(p_t) * log(p_t)
                        phi(p_t)  = (1 - p_t)^gamma / (1 + tau * (1 - p_t)^gamma)
- GFL-q (this paper):   GFL-q(p_t) = phi(p_t) * (1 - p_t^q) / q

All losses operate on raw logits and integer targets, use log_softmax for
numerical stability, support optional per-class weights `alpha` (shape (K,)),
and return a scalar loss (mean reduction) by default.
"""

from __future__ import annotations
from typing import Optional, Callable

import torch
import torch.nn.functional as F

__all__ = [
    "cross_entropy_loss",
    "focal_loss",
    "gce_loss",
    "mae_loss",
    "gfl_loss",
    "gfl_q_loss",
    "make_loss",
]


# ---------------------------------------------------------------------------
# Internal helper: gather log p_t from logits + integer targets
# ---------------------------------------------------------------------------
def _log_p_t(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return log p_t = log P(y | x) for the ground-truth class y, shape (N,).

    Uses log_softmax for numerical stability; never materialises p_t = exp(...)
    unless the calling loss explicitly needs it.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (N, K), got shape {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be 1-D (N,), got shape {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(f"batch size mismatch: logits {logits.size(0)} vs targets {targets.size(0)}")
    log_p = F.log_softmax(logits, dim=-1)
    return log_p.gather(1, targets.view(-1, 1)).squeeze(1)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction}")


def _apply_alpha(loss: torch.Tensor, targets: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
    if alpha is None:
        return loss
    if alpha.dim() != 1:
        raise ValueError(f"alpha must be 1-D (K,), got shape {tuple(alpha.shape)}")
    return alpha[targets] * loss


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------
def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """CE(p_t) = -log(p_t)."""
    log_p_t = _log_p_t(logits, targets)
    loss = -log_p_t
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# Focal loss (Lin et al., 2017)
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """FL(p_t) = -(1 - p_t)^gamma * log(p_t).  gamma >= 0; gamma=0 recovers CE."""
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    log_p_t = _log_p_t(logits, targets)
    p_t = log_p_t.exp().clamp(min=0.0, max=1.0)  # clamp guards float drift
    u = 1.0 - p_t
    modulator = u.pow(gamma)
    loss = -modulator * log_p_t
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# GCE (Zhang & Sabuncu, 2018)
# ---------------------------------------------------------------------------
def gce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    q: float = 0.7,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """GCE(p_t) = (1 - p_t^q) / q.  q in (0, 1]; q -> 0 recovers CE; q = 1 recovers MAE."""
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    log_p_t = _log_p_t(logits, targets)
    p_t = log_p_t.exp().clamp(min=0.0, max=1.0)
    loss = (1.0 - p_t.pow(q)) / q
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# MAE (multi-class form)
# ---------------------------------------------------------------------------
def mae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MAE(p_t) = 1 - p_t.  Symmetric, uniformly bounded; equivalent to GCE at q=1."""
    log_p_t = _log_p_t(logits, targets)
    p_t = log_p_t.exp().clamp(min=0.0, max=1.0)
    loss = 1.0 - p_t
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# GFL: Generalized Focal Loss   (this paper)
# ---------------------------------------------------------------------------
def gfl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    tau: float = 0.3,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Generalized Focal Loss.

        GFL(p_t) = -phi(p_t) * log(p_t)
        phi(p_t) = (1 - p_t)^gamma / (1 + tau * (1 - p_t)^gamma)

    Recovery: tau -> 0 yields focal loss; gamma -> 0 yields rescaled CE.
    Boundedness: phi(p_t) <= 1 / (1 + tau) on [0, 1].
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    log_p_t = _log_p_t(logits, targets)
    p_t = log_p_t.exp().clamp(min=0.0, max=1.0)
    u_g = (1.0 - p_t).pow(gamma)
    modulator = u_g / (1.0 + tau * u_g)
    loss = -modulator * log_p_t
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# GFL-q: q-deformed Generalized Focal Loss   (this paper, uniform B-robust)
# ---------------------------------------------------------------------------
def gfl_q_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    tau: float = 0.3,
    q: float = 0.7,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """q-deformed Generalized Focal Loss.

        GFL-q(p_t) = phi(p_t) * (1 - p_t^q) / q
        phi(p_t)   = (1 - p_t)^gamma / (1 + tau * (1 - p_t)^gamma)

    Replaces the natural log in GFL with the Tsallis q-logarithm.
    Achieves uniform bounded influence on [0, 1] for q in (0, 1].
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    log_p_t = _log_p_t(logits, targets)
    p_t = log_p_t.exp().clamp(min=0.0, max=1.0)
    u_g = (1.0 - p_t).pow(gamma)
    modulator = u_g / (1.0 + tau * u_g)
    deformed = (1.0 - p_t.pow(q)) / q
    loss = modulator * deformed
    loss = _apply_alpha(loss, targets, alpha)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# Loss factory: build a (logits, targets, alpha) -> scalar callable from a name
# ---------------------------------------------------------------------------
LossFn = Callable[..., torch.Tensor]


def make_loss(name: str, **kwargs) -> LossFn:
    """Return a loss callable with hyperparameters baked in.

    Usage
    -----
        loss_fn = make_loss("gfl", gamma=2.0, tau=0.3)
        loss = loss_fn(logits, targets, alpha=class_weights)
    """
    name = name.lower()
    if name == "ce":
        return lambda logits, targets, alpha=None: cross_entropy_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    if name == "fl":
        return lambda logits, targets, alpha=None: focal_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    if name == "gce":
        return lambda logits, targets, alpha=None: gce_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    if name == "mae":
        return lambda logits, targets, alpha=None: mae_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    if name == "gfl":
        return lambda logits, targets, alpha=None: gfl_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    if name == "gfl_q":
        return lambda logits, targets, alpha=None: gfl_q_loss(
            logits, targets, alpha=alpha, **kwargs
        )
    raise ValueError(f"unknown loss name: {name!r}. expected one of: ce, fl, gce, mae, gfl, gfl_q")


# ===========================================================================
# Tests   —   run with: python losses.py
# ===========================================================================
def _run_tests() -> None:
    """Smoke tests + recovery-limit checks. All must pass."""
    torch.manual_seed(0)

    K, N = 5, 64
    logits = torch.randn(N, K, requires_grad=True)
    targets = torch.randint(0, K, (N,))

    # --- 1. Every loss returns finite, non-negative scalars with finite grads.
    for name in ["ce", "fl", "gce", "mae", "gfl", "gfl_q"]:
        if logits.grad is not None:
            logits.grad.zero_()
        loss_fn = make_loss(name)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all(), f"{name}: non-finite loss value: {loss.item()}"
        assert loss.item() >= 0.0, f"{name}: negative loss: {loss.item()}"
        loss.backward()
        assert torch.isfinite(logits.grad).all(), f"{name}: non-finite gradient"
    print("  [pass] all losses give finite non-negative values + finite grads")

    # --- 2. Recovery: GFL with tau=0 must equal FL exactly.
    fl_val = focal_loss(logits, targets, gamma=2.0)
    gfl_recover = gfl_loss(logits, targets, gamma=2.0, tau=0.0)
    assert torch.allclose(fl_val, gfl_recover, atol=1e-6, rtol=1e-6), \
        f"GFL(tau=0) != FL: {fl_val.item():.8f} vs {gfl_recover.item():.8f}"
    print(f"  [pass] GFL(tau=0) recovers FL exactly: {fl_val.item():.6f} == {gfl_recover.item():.6f}")

    # --- 3. Recovery: FL with gamma=0 must equal CE exactly.
    ce_val = cross_entropy_loss(logits, targets)
    fl_recover = focal_loss(logits, targets, gamma=0.0)
    assert torch.allclose(ce_val, fl_recover, atol=1e-6, rtol=1e-6), \
        f"FL(gamma=0) != CE: {ce_val.item():.8f} vs {fl_recover.item():.8f}"
    print(f"  [pass] FL(gamma=0) recovers CE exactly: {ce_val.item():.6f} == {fl_recover.item():.6f}")

    # --- 4. Recovery: GFL with tau=0 AND gamma=0 must equal CE exactly.
    gfl_to_ce = gfl_loss(logits, targets, gamma=0.0, tau=0.0)
    assert torch.allclose(ce_val, gfl_to_ce, atol=1e-6, rtol=1e-6), \
        f"GFL(0,0) != CE: {ce_val.item():.8f} vs {gfl_to_ce.item():.8f}"
    print(f"  [pass] GFL(gamma=0, tau=0) recovers CE: {ce_val.item():.6f} == {gfl_to_ce.item():.6f}")

    # --- 5. Boundedness: phi(p_t) <= 1/(1+tau) for any p_t in [0,1].
    tau = 0.5
    p_grid = torch.linspace(0.0, 1.0, 1001)
    u_g = (1.0 - p_grid).pow(2.0)
    phi_vals = u_g / (1.0 + tau * u_g)
    bound = 1.0 / (1.0 + tau)
    assert phi_vals.max().item() <= bound + 1e-6, \
        f"phi exceeds 1/(1+tau): max={phi_vals.max():.6f}, bound={bound:.6f}"
    assert abs(phi_vals[0].item() - bound) < 1e-6, \
        f"phi(p=0) should equal 1/(1+tau): got {phi_vals[0].item():.6f}, want {bound:.6f}"
    print(f"  [pass] modulator bound: max phi = {phi_vals.max().item():.6f} <= 1/(1+tau) = {bound:.6f}")

    # --- 6. Edge: extremely confident correct prediction => near-zero loss.
    confident_logits = torch.tensor([[10.0, -10.0, -10.0]])
    target = torch.tensor([0])
    for name in ["ce", "fl", "gce", "mae", "gfl", "gfl_q"]:
        loss_fn = make_loss(name)
        v = loss_fn(confident_logits, target).item()
        assert torch.isfinite(torch.tensor(v)), f"{name}: NaN on confident-correct"
        assert v < 0.1, f"{name}: loss {v:.4f} too high on confident-correct"
    print("  [pass] confident-correct prediction => small loss for all losses")

    # --- 7. Edge: confidently WRONG prediction => loss is finite (this is the noise case).
    wrong_logits = torch.tensor([[-50.0, 50.0, -50.0]])
    target = torch.tensor([0])
    losses_wrong = {}
    for name in ["ce", "fl", "gce", "mae", "gfl", "gfl_q"]:
        loss_fn = make_loss(name)
        v = loss_fn(wrong_logits, target).item()
        assert torch.isfinite(torch.tensor(v)), f"{name}: non-finite loss on confidently-wrong: {v}"
        losses_wrong[name] = v
    print("  [pass] confidently-wrong predictions give finite loss for all losses:")
    for name, v in losses_wrong.items():
        print(f"           {name:>6s}: {v:>10.4f}")
    # GFL should be < FL on confidently-wrong (bounded-influence prediction).
    assert losses_wrong["gfl"] < losses_wrong["fl"], \
        f"expected GFL < FL on confidently-wrong, got GFL={losses_wrong['gfl']:.4f} >= FL={losses_wrong['fl']:.4f}"
    # GFL-q should be < GFL (the q-deformation tightens the bound).
    assert losses_wrong["gfl_q"] < losses_wrong["gfl"], \
        f"expected GFL-q < GFL on confidently-wrong, got GFL-q={losses_wrong['gfl_q']:.4f} >= GFL={losses_wrong['gfl']:.4f}"
    print(f"  [pass] bounded-influence ordering: GFL-q ({losses_wrong['gfl_q']:.4f}) < "
          f"GFL ({losses_wrong['gfl']:.4f}) < FL ({losses_wrong['fl']:.4f})")

    # --- 8. Per-class alpha works.
    alpha = torch.tensor([2.0, 0.5, 1.0, 1.0, 1.0])
    base = cross_entropy_loss(logits, targets)
    weighted = cross_entropy_loss(logits, targets, alpha=alpha)
    assert torch.isfinite(weighted).all() and weighted.item() != base.item()
    print("  [pass] per-class alpha weighting")

    print("\nAll loss tests passed.")


if __name__ == "__main__":
    print("Running losses.py self-tests...\n")
    _run_tests()
