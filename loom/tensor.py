import numpy as np


def unbroadcast(grad, target_shape):

        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)

        for i, (g,t) in enumerate(zip(grad.shape, target_shape)):
            if t==1 and g!=1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad 

class Tensor:

    def __init__(self, data, parents=(), op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda :None
        self._prev = set(parents)
        # self.requires_grad = requires_grad
        self._op = op

    @property
    def shape(self): return self.data.shape

    @property
    def dtype(self): return self.data.dtype

    @property
    def ndim(self): return self.data.ndim

    @property
    def size(self): return self.data.size

    # def device(self):
    #     return self._device TODO

    def __repr__(self):

        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor(shape={self.shape!r}{op})"


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out 

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad +=  unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "exponent must be a plain number"

        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += unbroadcast(other * ((self.data)**(other - 1)) * out.grad, self.shape)
        out._backward = _backward

        return out
    
    def __neg__(self): return self * -1

    def __sub__(self, other):  return self + -(other)

    def __truediv__(self, other): return self * other ** -1

    def __rmul__(self, other): return self * other

    def __radd__(self, other): return self + other 

    def __rtruediv__(self, other): return other * self ** -1

    def __rsub__(self, other): return (-self) + other

    
    def exp(self):
        x = self.data
        out = Tensor(np.exp(x), (self,), 'exp' )
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += 1/self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):

        A = self.data

        out = Tensor(np.maximum(A , 0), (self,), 'relu')
        def _backward():
            self.grad += out.grad * (A >0)
        out._backward = _backward

        return out

    def sum(self, axis = None, keepdims = False):

        out = Tensor(self.data.sum(axis = axis, keepdims = keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis) #puttin vanished axes back 
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = _backward

        return out

    def mean(self, axis = None, keepdims= False):
        s = self.sum(axis = axis, keepdims = keepdims)
        count = self.data.size // s.data.size 

        out = s / count 

        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def transpose(self, *axes):
        axes = axes if axes else None 
        out = Tensor(np.transpose(self.data, axes), (self,), 'transpose')
        def _backward():
            inv = None if axes is None else tuple(np.argsort(axes))
            self.grad += np.transpose(out.grad, inv)
        out._backward = _backward

        return out


    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


