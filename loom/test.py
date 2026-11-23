
import numpy as np
from loom.tensor import Tensor


# ============================================================
# TEST RUNNER
# ============================================================

passed = 0
failed = 0


def check(name, condition):
    global passed, failed

    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


def check_close(name, actual, expected, atol=1e-5):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    check(
        name,
        actual.shape == expected.shape
        and np.allclose(actual, expected, atol=atol)
    )


def section(name):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)


# ============================================================
# NUMERICAL GRADIENT CHECK
# ============================================================

def numerical_gradient(fn, x, eps=1e-6):
    """
    Computes gradients using central finite differences.

    fn: function that accepts a NumPy array and returns
        a scalar Tensor.
    x: input NumPy array.
    """

    grad = np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[idx] += eps
        x_minus[idx] -= eps

        y_plus = fn(x_plus).data.item()
        y_minus = fn(x_minus).data.item()

        grad[idx] = (y_plus - y_minus) / (2 * eps)

    return grad


def gradcheck(name, fn, x, atol=1e-4):
    """
    Compares analytical gradients from your engine
    against numerical finite-difference gradients.
    """

    x_tensor = Tensor(x.copy())

    output = fn(x_tensor)

    if output.data.size != 1:
        output = output.sum()

    output.backward()

    analytical = x_tensor.grad.copy()
    numerical = numerical_gradient(
        lambda arr: fn(Tensor(arr)).sum(),
        x.copy()
    )

    check_close(name, analytical, numerical, atol=atol)


# ============================================================
# 1. CONSTRUCTION AND PROPERTIES
# ============================================================

section("1. CONSTRUCTION AND PROPERTIES")

x = Tensor([[1, 2, 3], [4, 5, 6]])

check_close("Data", x.data, [[1, 2, 3], [4, 5, 6]])
check("Shape", x.shape == (2, 3))
check("NDIM", x.ndim == 2)
check("Size", x.size == 6)
check("Dtype", x.dtype == np.float64)
check_close("Initial gradient", x.grad, np.zeros((2, 3)))


# ============================================================
# 2. ADDITION
# ============================================================

section("2. ADDITION")

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

c = a + b

check_close("Forward addition", c.data, [5, 7, 9])

c.sum().backward()

check_close("Addition gradient - left", a.grad, [1, 1, 1])
check_close("Addition gradient - right", b.grad, [1, 1, 1])


# Broadcasting
a = Tensor([[1, 2, 3], [4, 5, 6]])
b = Tensor([10, 20, 30])

c = a + b

check_close(
    "Broadcasted addition",
    c.data,
    [[11, 22, 33], [14, 25, 36]]
)

c.sum().backward()

check_close("Broadcast gradient - matrix", a.grad, np.ones((2, 3)))
check_close("Broadcast gradient - vector", b.grad, [2, 2, 2])


# ============================================================
# 3. MULTIPLICATION
# ============================================================

section("3. MULTIPLICATION")

a = Tensor([2, 3, 4])
b = Tensor([5, 6, 7])

c = a * b

check_close("Forward multiplication", c.data, [10, 18, 28])

c.sum().backward()

check_close("Multiplication gradient - left", a.grad, [5, 6, 7])
check_close("Multiplication gradient - right", b.grad, [2, 3, 4])


# Broadcasting multiplication
a = Tensor([[1, 2, 3], [4, 5, 6]])
b = Tensor([2, 3, 4])

c = a * b
c.sum().backward()

check_close(
    "Broadcast multiplication gradient - matrix",
    a.grad,
    [[2, 3, 4], [2, 3, 4]]
)

check_close(
    "Broadcast multiplication gradient - vector",
    b.grad,
    [5, 7, 9]
)


# ============================================================
# 4. POWER, NEGATION, SUBTRACTION, DIVISION
# ============================================================

section("4. ARITHMETIC OPERATIONS")

x = Tensor([2, 3, 4])
y = x ** 3

check_close("Power forward", y.data, [8, 27, 64])

y.sum().backward()

check_close("Power gradient", x.grad, [12, 27, 48])


x = Tensor([1, 2, 3])
y = -x

check_close("Negation forward", y.data, [-1, -2, -3])


a = Tensor([5, 7, 9])
b = Tensor([2, 3, 4])

c = a - b

check_close("Subtraction forward", c.data, [3, 4, 5])


a = Tensor([10, 20, 30])
b = Tensor([2, 4, 5])

c = a / b

check_close("Division forward", c.data, [5, 5, 6])


# Reflected operations
x = Tensor([2, 4, 5])

check_close("Reflected addition", (10 + x).data, [12, 14, 15])
check_close("Reflected multiplication", (3 * x).data, [6, 12, 15])
check_close("Reflected subtraction", (10 - x).data, [8, 6, 5])
check_close("Reflected division", (10 / x).data, [5, 2.5, 2])


# ============================================================
# 5. EXP, LOG, RELU
# ============================================================

section("5. ACTIVATION AND MATH OPERATIONS")

x = Tensor([0, 1, 2])
y = x.exp()

check_close("Exp forward", y.data, np.exp([0, 1, 2]))

y.sum().backward()

check_close("Exp gradient", x.grad, np.exp([0, 1, 2]))


x = Tensor([1, np.e, np.e ** 2])
y = x.log()

check_close("Log forward", y.data, [0, 1, 2])

y.sum().backward()

check_close("Log gradient", x.grad, 1 / x.data)


x = Tensor([-2, -1, 0, 1, 2])
y = x.relu()

check_close("ReLU forward", y.data, [0, 0, 0, 1, 2])

y.sum().backward()

check_close("ReLU gradient", x.grad, [0, 0, 0, 1, 1])


# ============================================================
# 6. SUM AND MEAN
# ============================================================

section("6. REDUCTIONS")

x = Tensor([[1, 2, 3], [4, 5, 6]])

check_close("Sum all elements", x.sum().data, 21)
check_close("Sum axis 0", x.sum(axis=0).data, [5, 7, 9])
check_close("Sum axis 1", x.sum(axis=1).data, [6, 15])

check_close(
    "Sum axis 0 keepdims",
    x.sum(axis=0, keepdims=True).data,
    [[5, 7, 9]]
)

check_close(
    "Sum axis 1 keepdims",
    x.sum(axis=1, keepdims=True).data,
    [[6], [15]]
)


# Sum backward: all elements
x = Tensor([[1, 2, 3], [4, 5, 6]])
x.sum().backward()

check_close("Sum gradient - all", x.grad, np.ones((2, 3)))


# Sum backward: axis 0
x = Tensor([[1, 2, 3], [4, 5, 6]])
x.sum(axis=0).sum().backward()

check_close("Sum gradient - axis 0", x.grad, np.ones((2, 3)))


# Sum backward: axis 1
x = Tensor([[1, 2, 3], [4, 5, 6]])
x.sum(axis=1).sum().backward()

check_close("Sum gradient - axis 1", x.grad, np.ones((2, 3)))


# Mean
x = Tensor([[1, 2, 3], [4, 5, 6]])

check_close("Mean all", x.mean().data, 3.5)
check_close("Mean axis 0", x.mean(axis=0).data, [2.5, 3.5, 4.5])
check_close("Mean axis 1", x.mean(axis=1).data, [2, 5])

x.mean().backward()

check_close(
    "Mean gradient",
    x.grad,
    np.full((2, 3), 1 / 6)
)


# ============================================================
# 7. RESHAPE
# ============================================================

section("7. RESHAPE")

x = Tensor(np.arange(6))

y = x.reshape(2, 3)

check("Reshape shape", y.shape == (2, 3))

check_close(
    "Reshape forward",
    y.data,
    [[0, 1, 2], [3, 4, 5]]
)

y.sum().backward()

check_close("Reshape gradient", x.grad, np.ones(6))


# ============================================================
# 8. TRANSPOSE
# ============================================================

section("8. TRANSPOSE")

x = Tensor([[1, 2, 3], [4, 5, 6]])

y = x.transpose(1, 0)

check("Transpose shape", y.shape == (3, 2))

check_close(
    "Transpose forward",
    y.data,
    [[1, 4], [2, 5], [3, 6]]
)

y.sum().backward()

check_close("Transpose gradient", x.grad, np.ones((2, 3)))


# 3D transpose
x = Tensor(np.arange(24).reshape(2, 3, 4))

y = x.transpose(1, 0, 2)

check("3D transpose shape", y.shape == (3, 2, 4))

y.sum().backward()

check_close("3D transpose gradient", x.grad, np.ones((2, 3, 4)))


# ============================================================
# 9. SHARED COMPUTATION GRAPHS
# ============================================================

section("9. COMPUTATION GRAPH AND GRADIENT ACCUMULATION")

x = Tensor([2, 3])

# Same Tensor used in multiple branches
y = x * x + x

check_close("Shared graph forward", y.data, [6, 12])

y.sum().backward()

# d(x*x + x)/dx = 2x + 1
check_close("Shared graph gradient", x.grad, [5, 7])


# Another shared graph
x = Tensor([2, 3])

a = x * 2
b = x * 3
y = a + b

y.sum().backward()

check_close("Multi-branch gradient accumulation", x.grad, [5, 5])


# ============================================================
# 10. NUMERICAL GRADIENT CHECKS
# ============================================================

section("10. NUMERICAL GRADIENT CHECKS")

# Addition
gradcheck(
    "Gradcheck: addition",
    lambda x: (x + 3).sum(),
    np.array([1.0, 2.0, 3.0])
)

# Multiplication
gradcheck(
    "Gradcheck: multiplication",
    lambda x: (x * x).sum(),
    np.array([1.0, 2.0, 3.0])
)

# Power
gradcheck(
    "Gradcheck: power",
    lambda x: (x ** 3).sum(),
    np.array([1.0, 2.0, 3.0])
)

# Exponential
gradcheck(
    "Gradcheck: exp",
    lambda x: x.exp().sum(),
    np.array([0.1, 0.5, 1.0])
)

# Logarithm
gradcheck(
    "Gradcheck: log",
    lambda x: x.log().sum(),
    np.array([1.0, 2.0, 3.0])
)

# ReLU: avoid the nondifferentiable point at zero
gradcheck(
    "Gradcheck: ReLU",
    lambda x: x.relu().sum(),
    np.array([-2.0, -1.0, 1.0, 2.0])
)

# Broadcasting addition
gradcheck(
    "Gradcheck: broadcast addition",
    lambda x: (x + Tensor([1.0, 2.0, 3.0])).sum(),
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
)

# Broadcasting multiplication
gradcheck(
    "Gradcheck: broadcast multiplication",
    lambda x: (x * Tensor([2.0, 3.0, 4.0])).sum(),
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
)

# Mean
gradcheck(
    "Gradcheck: mean",
    lambda x: x.mean(),
    np.array([[1.0, 2.0], [3.0, 4.0]])
)

# Reshape
gradcheck(
    "Gradcheck: reshape",
    lambda x: x.reshape(2, 2).sum(),
    np.array([1.0, 2.0, 3.0, 4.0])
)

# Transpose
gradcheck(
    "Gradcheck: transpose",
    lambda x: x.transpose(1, 0).sum(),
    np.array([[1.0, 2.0], [3.0, 4.0]])
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total = passed + failed

print("\n" + "=" * 55)
print("TEST SUMMARY")
print("=" * 55)

print(f"Total tests : {total}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")

if failed == 0:
    print("\nALL TESTS PASSED!")
else:
    print(f"\n{failed} TEST(S) FAILED.")

print("=" * 55)
