from experiments.grad_engine import Value
from experiments.nn import MLP

# x1 = Value(2.0); x2 = Value(0.0)
# w1 = Value(-3.0); w2 = Value(1.0)
# b = Value(6.8813735870195432)
# o = (x1*w1 + x2*w2 + b).tanh()
# o.backward()
# print(x1.grad, w1.grad, x2.grad, w2.grad)   # expect -1.5, 1.0, 0.5, 0.0

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, -1.0]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,4,1])
n(x)

for k in range(1000):

    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.005 * p.grad


print(k, loss.data)
print(ypred)
