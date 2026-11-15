import numpy as np
from loom.tensor import Tensor


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def attempt(name, fn):
    """Run a check that may crash (e.g. a shape error) and report it instead of stopping."""
    try:
        check(name, fn())
    except Exception as e:
        print(f"FAIL  {name}  ->  {type(e).__name__}: {e}")


a = Tensor([[1, 2, 3], [4, 5, 6]])    # shape (2, 3)
b = Tensor([10, 20, 30])              # shape (3,)
c = Tensor([[100], [200]])            # shape (2, 1)

print("--- construction and repr")
print("repr of a    :", repr(a))
print("repr of a + b:", repr(a + b))
check("leaf repr has no op", repr(a) == "Tensor(shape=(2, 3))")
check("result repr shows op='+'", "op='+'" in repr(a + b))
check("float64", a.data.dtype == np.float64)
check("grad is zeros with data's shape", a.grad.shape == (2, 3) and np.all(a.grad == 0))
check("shape / ndim / size", (a.shape, a.ndim, a.size) == ((2, 3), 2, 6))

print("--- add forward")
check("same shape", np.allclose((a + a).data, [[2, 4, 6], [8, 10, 12]]))
check("(2,3) + (3,)", np.allclose((a + b).data, [[11, 22, 33], [14, 25, 36]]))
check("(2,3) + (2,1)", np.allclose((a + c).data, [[101, 102, 103], [204, 205, 206]]))
check("python number on the right", np.allclose((a + 5).data, [[6, 7, 8], [9, 10, 11]]))
check("inputs not modified", np.array_equal(a.data, [[1, 2, 3], [4, 5, 6]])
      and np.array_equal(b.data, [10, 20, 30]))
out = a + b
check("parents recorded", out._prev == {a, b})
if hasattr(Tensor, "__radd__"):
    check("python number on the left", np.allclose((5 + a).data, [[6, 7, 8], [9, 10, 11]]))
else:
    print("SKIP  python number on the left (__radd__ not written yet)")

print("--- add backward")
if not hasattr(Tensor, "backward"):
    print("SKIP  backward() not written yet")
else:
    def same_shape():
        p = Tensor([[1.0, 2.0], [3.0, 4.0]])
        q = Tensor([[5.0, 6.0], [7.0, 8.0]])
        (p + q).backward()
        return np.array_equal(p.grad, np.ones((2, 2))) and np.array_equal(q.grad, np.ones((2, 2)))

    def used_twice():
        x = Tensor([1.0, 2.0, 3.0])
        (x + x).backward()
        return np.array_equal(x.grad, [2.0, 2.0, 2.0])

    def chain():
        p, q, r = Tensor([1.0]), Tensor([2.0]), Tensor([3.0])
        ((p + q) + r).backward()
        return p.grad[0] == 1.0 and q.grad[0] == 1.0 and r.grad[0] == 1.0

    def row_bias():  # (2,3) + (3,)
        p = Tensor([[1, 2, 3], [4, 5, 6]])
        q = Tensor([10, 20, 30])
        (p + q).backward()
        return (np.array_equal(p.grad, np.ones((2, 3)))
                and q.grad.shape == (3,)
                and np.array_equal(q.grad, [2.0, 2.0, 2.0]))

    def column():  # (2,3) + (2,1)
        p = Tensor(np.zeros((2, 3)))
        q = Tensor([[100], [200]])
        (p + q).backward()
        return q.grad.shape == (2, 1) and np.array_equal(q.grad, [[3.0], [3.0]])

    def scalar():  # (2,3) + ()
        p = Tensor(np.zeros((2, 3)))
        q = Tensor(5.0)
        (p + q).backward()
        return np.array_equal(p.grad, np.ones((2, 3))) and np.allclose(q.grad, 6.0)

    attempt("same shape: both grads are ones", same_shape)
    attempt("x + x: grad accumulates to 2", used_twice)
    attempt("(a + b) + c: all grads are 1", chain)
    attempt("(2,3) + (3,):  b.grad == [2, 2, 2]", row_bias)
    attempt("(2,3) + (2,1): c.grad == [[3], [3]]", column)
    attempt("(2,3) + scalar: s.grad == 6", scalar)