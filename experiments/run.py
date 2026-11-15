from experiments.nn import MLP


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

# x = [2.0, 3.0, -1.0]

n = MLP(3, [4,4,1])

# n(x)

for k in range(50):

    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.5 * p.grad


    print(k, loss.data)

print(ypred)