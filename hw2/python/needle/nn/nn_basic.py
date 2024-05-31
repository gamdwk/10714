"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=bias).reshape((1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        h = X @ self.weight
        if self.bias.requires_grad:
            h = h + ops.broadcast_to(self.bias, h.shape)
        return h
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = 1
        for s in X.shape[1:]:
            shape = shape * s
        h = X.reshape((X.shape[0],  shape))
        # print(h.shape)
        return h
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        h = x
        for model in self.modules:
            h = model(h)
        return h
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[0]
        # get index = y[i] in I.
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        #x = ops.exp(logits).sum(1)
        #y = ops.log(x).sum()
        y = ops.logsumexp(logits, axes=(1,)).sum()
        z = (logits * y_one_hot).sum()
        loss = y - z
        return loss / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype,  requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            ex = x.sum(0) / x.shape[0] # dim,
            self.running_mean.data = (1- self.momentum) * self.running_mean.data + ex.data * self.momentum
            ex = ops.broadcast_to(ex, x.shape) # batch_size, dim
            sub = x - ex
            var_x = ops.summation(sub * sub, axes=0) / x.shape[0]
            self.running_var.data = (1 - self.momentum) * self.running_var.data + var_x.data * self.momentum
            y = (var_x + self.eps) ** 0.5
            y = ops.broadcast_to(y, x.shape)
            w = ops.broadcast_to(self.weight, x.shape)
            b = ops.broadcast_to(self.bias, x.shape)
            h = w * (sub / y) + b
        else:
            ex = ops.broadcast_to(self.running_mean, x.shape) # batch_size, dim
            var_x = self.running_var
            sub = x - ex
            y = (var_x + self.eps) ** 0.5
            y = ops.broadcast_to(y, x.shape)
            w = ops.broadcast_to(self.weight, x.shape)
            b = ops.broadcast_to(self.bias, x.shape)
            h = w * (sub / y) + b

        return h
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ex = x.sum(1) / x.shape[1] # batch_size, 
        ex = ex.reshape((x.shape[0], 1))
        ex = ops.broadcast_to(ex, x.shape) # batch_size, dim
        sub = x - ex
        
        var_x =  ops.summation(sub * sub, axes=1) / x.shape[1]
        y = (var_x + self.eps) ** 0.5
        
        y = ops.broadcast_to(y.reshape((y.shape[0], 1)), x.shape)
        w = self.weight
        w = ops.broadcast_to(w.reshape((1, w.shape[0])), x.shape)
        b = self.bias
        b = ops.broadcast_to(b.reshape((1, b.shape[0])), x.shape)
        
        return w * (sub / y) + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is False:
            return x
        y = init.randb(*x.shape, p=1-self.p)
        return x * y / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
