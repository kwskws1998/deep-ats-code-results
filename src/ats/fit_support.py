"""Numerical fit selection that retains failed solutions and only accepts equivalent converged refinements."""

import numpy as np
from scipy.optimize import minimize


def select_optimization(results, objective, bounds, args=(), jac=False):
    """Polish an unconverged optimum without silently replacing it by a materially worse converged fit."""
    finite = [
        result
        for result in results
        if np.isfinite(result.fun) and np.isfinite(result.x).all()
    ]
    if not finite:
        raise FloatingPointError("No finite likelihood optimization result.")
    best = min(finite, key=lambda result: result.fun)
    if best.success:
        return best

    def scalar(parameters):
        """Provide the same objective to a derivative-free numerical fallback."""
        result = objective(parameters, *args)
        return result[0] if jac else result

    fallback = minimize(
        scalar,
        best.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "xtol": 1e-7},
    )
    refined = minimize(
        objective,
        fallback.x,
        args=args,
        jac=jac,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7},
    )
    candidates = finite + [
        result
        for result in (fallback, refined)
        if np.isfinite(result.fun) and np.isfinite(result.x).all()
    ]
    best = min(candidates, key=lambda result: result.fun)
    tolerance = 1e-7 * max(1, abs(best.fun))
    converged = [
        result
        for result in candidates
        if result.success and result.fun <= best.fun + tolerance
    ]
    return min(converged, key=lambda result: result.fun) if converged else best
