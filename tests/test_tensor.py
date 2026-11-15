import numpy as np
from tapeloom.tensor import Tensor

def test_tensor_init():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.data.dtype == np.float64
    assert t.grad.shape == (2, 3)
    assert np.all(t.grad == 0)