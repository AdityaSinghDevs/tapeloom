"""Plain-Python checks for every Tensor op (no pytest needed).

Run from the repo root:    python tests/check_ops.py
Prints PASS or FAIL for each case, then a summary. Exit code is 1 if anything fails.

Two kinds of checks:
  * "gradcheck": compares Tensor.backward() with finite differences
    (this proves the gradient matches the function the op computes).
  * "value": checks forward results and exact gradients against numbers worked out by hand
    (this proves the op computes the RIGHT function; a gradcheck alone cannot,
    because a wrong function with a matching gradient still passes it).
"""
import os
import sys

# make `import loom` work when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from loom.tensor import Tensor, unbroadcast

RNG = np.random.default_rng(0)      # seeded, so runs are repeatable
RESULTS = []


def attempt(name, fn):
    """Run fn() -> bool. An exception counts as a FAIL and is shown."""
    try:
        ok = bool(fn())
        note = ""
    except Exception as e:
        ok = False
        note = f"   <- {type(e).__name__}: {e}"
    RESULTS.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + note)


def same(a, b):
    """Same shape AND same numbers."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b)


# ---------------------------------------------------------------- gradcheck helpers
def numerical_grad(f, x, eps=1e-6):
    """Central difference: slope of the scalar function f at array x, one element at a time."""
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps
        plus = f(x)
        x[idx] = old - eps
        minus = f(x)
        x[idx] = old
        g[idx] = (plus - minus) / (2 * eps)
    return g


def gradcheck(name, op, *inputs):
    """Compare backward() with finite differences for op(*Tensors) -> Tensor."""
    def run():
        arrays = [np.array(a, dtype=np.float64) for a in inputs]
        out_shape = op(*[Tensor(a.copy()) for a in arrays]).shape
        # random weights, so every output element gets a different incoming gradient
        w = RNG.normal(size=out_shape)

        def scalar(arrs):
            return float((op(*[Tensor(a.copy()) for a in arrs]).data * w).sum())

        tensors = [Tensor(a.copy()) for a in arrays]
        (op(*tensors) * Tensor(w)).sum().backward()

        for i, t in enumerate(tensors):
            def f(x, i=i):
                return scalar([x if j == i else arrays[j] for j in range(len(arrays))])
            num = numerical_grad(f, arrays[i])
            if t.grad.shape != num.shape or not np.allclose(t.grad, num, atol=1e-6, rtol=1e-4):
                return False
        return True
    attempt("gradcheck  " + name, run)


# ---------------------------------------------------------------- test data
A = [[1., 2., 3.], [4., 5., 6.]]             # (2,3)
B = [[0.5, -1., 2.], [1.5, 3., -2.]]         # (2,3)
B2 = [[0.5, 1., 2.], [1.5, 3., -2.]]         # (2,3)