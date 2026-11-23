from loom.tensor import Tensor

x = Tensor([1.0])
y = x
for _ in range(5000):
    y = y + x          # builds a chain 5000 nodes deep
y.backward()
print(x.grad)          # expect [5001.]

x = Tensor([2., 3.])
(x * x).sum().backward()
print(x.grad)                    # [4. 6.]
x.zero_grad()
print(x.grad)                    # [0. 0.]
print(Tensor(3.14).item())       # 3.14
a = x.numpy(); a[0] = 99
print(x.data)                    # still [2. 3.], because numpy() returns a copy