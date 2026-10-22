"""Two-dimensional SPD algebra without a host synchronization per matrix.

These formulas retain the reference NLL jitter and symmetric SPD derivatives.
Invalid pivots produce nonfinite values, checked at the update/rollout boundary;
no variance clipping or silent covariance repair changes the objective.
"""

import math

import torch


def cholesky2(matrix):
    """Factor [...,2,2] SPD matrices with differentiable scalar tensor operations."""
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("Expected two-dimensional covariance matrices.")
    first = matrix[..., 0, 0].sqrt()
    cross = (0.5 * (matrix[..., 1, 0] + matrix[..., 0, 1])) / first
    second = (matrix[..., 1, 1] - cross.square()).sqrt()
    return first, cross, second


def solve_spd2(matrix, rhs):
    """Solve [...,2,2] @ X = [...,2,K] without synchronous linalg error checks."""
    first, cross, second = cholesky2(matrix)
    upper = rhs[..., 0, :] / first[..., None]
    lower = (rhs[..., 1, :] - cross[..., None] * upper) / second[..., None]
    bottom = lower / second[..., None]
    top = (upper - cross[..., None] * bottom) / first[..., None]
    return torch.stack((top, bottom), -2)


def gaussian_nll2(target, mean, covariance):
    """Match gaussian_nll for 2D states, including its explicit 1e-7 jitter."""
    identity = torch.eye(2, device=mean.device, dtype=mean.dtype)
    first, cross, second = cholesky2(covariance + 1e-7 * identity)
    error = target - mean
    upper = error[..., 0] / first
    lower = (error[..., 1] - cross * upper) / second
    return (
        math.log(2 * math.pi)
        + first.log()
        + second.log()
        + 0.5 * (upper.square() + lower.square())
    )
