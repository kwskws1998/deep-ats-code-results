"""Diagnostic visibility and Gaussian-position calibration with explicit evaluation masks."""

from __future__ import annotations

import math
import numpy as np


def visibility_calibration(probability, visible, mask) -> dict:
    """Report Brier/ECE and threshold-0.5 confusion counts on scored frames only."""
    p = np.asarray(probability)[mask].astype(float)
    truth = np.asarray(visible)[mask].astype(bool)
    if not len(p):
        return dict(visibility_samples=0)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Visibility probabilities must be finite in [0,1].")
    predicted = p >= 0.5
    tp, tn = int((predicted & truth).sum()), int((~predicted & ~truth).sum())
    fp, fn = int((predicted & ~truth).sum()), int((~predicted & truth).sum())
    ece = 0.0
    bins = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(p[selected].mean() - truth[selected].mean())
    return dict(
        visibility_samples=len(p),
        visibility_brier=float(np.mean((p - truth) ** 2)),
        visibility_ece10=float(ece),
        visibility_tp=tp,
        visibility_tn=tn,
        visibility_fp=fp,
        visibility_fn=fn,
    )


def position_calibration(target, mean, covariance, mask) -> dict:
    """Evaluate 2D Gaussian coverage using chi-square(2) thresholds and declared jitter."""
    error = (np.asarray(target) - mean)[mask].astype(float)
    cov = np.asarray(covariance)[mask].astype(float)
    if not len(error):
        return dict(position_samples=0)
    if not np.isfinite(error).all() or not np.isfinite(cov).all():
        raise FloatingPointError("Nonfinite position/covariance in diagnostic trace.")
    asymmetry = float(np.max(np.abs(cov - cov.swapaxes(-1, -2))))
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() <= -1e-7 or asymmetry > 1e-5:
        raise FloatingPointError(
            "Position covariance is not symmetric positive semidefinite."
        )
    regularized = cov + 1e-7 * np.eye(2)
    factor = np.linalg.cholesky(regularized)
    whitened = np.linalg.solve(factor, error[..., None])[..., 0]
    mahalanobis = np.sum(whitened**2, -1)
    nll = (
        math.log(2 * math.pi)
        + np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(-1)
        + 0.5 * mahalanobis
    )
    return dict(
        position_samples=len(error),
        position_nll=float(nll.mean()),
        mean_mahalanobis_squared=float(mahalanobis.mean()),
        coverage68=float(np.mean(mahalanobis <= -2 * math.log(1 - 0.68))),
        coverage95=float(np.mean(mahalanobis <= -2 * math.log(1 - 0.95))),
        minimum_covariance_eigenvalue=float(eigenvalues.min()),
        maximum_covariance_asymmetry=asymmetry,
    )
