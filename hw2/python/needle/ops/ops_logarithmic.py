from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..init import one_hot

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTIONs
        max_z = array_api.max(Z, axis=1).reshape(Z.shape[0], 1)
        Z_off = Z - max_z
        y = array_api.exp(Z_off)
        y = array_api.sum(y , axis=1)
        y = array_api.log(y).reshape(Z.shape[0], 1)
        return Z_off - y
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        # max_z = array_api.max(Z.realize_cached_data(), axis=1).reshape((Z.shape[0], 1))
        # x = Z - array_api.broadcast_to(max_z, Z.shape)
        '''
        y = exp(x)
        
        y = y / broadcast_to(sum_y, out_grad.shape)
        
        I = array_api.zeros(y.shape)
        I[array_api.arange(y.shape[0]), array_api.arange(y.shape[0])] = 1
        I = Tensor(I)
        grad = out_grad * y
        '''
        y = exp(Z)
        grad = summation(out_grad, axes=1).reshape((Z.shape[0], 1))
        sum_y = summation(y, axes=1).reshape((Z.shape[0], 1))
        grad = grad / sum_y
        grad = broadcast_to(grad, Z.shape)
        grad = y * grad
        return out_grad - grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes)
        if self.axes is None:
            shape = numpy.ones(len(Z.shape)).astype(int)
        else:
            shape = numpy.array(Z.shape)
            shape[list(self.axes)] = 1
        broadcast_max_z = numpy.broadcast_to(numpy.reshape(max_z,shape), Z.shape)
        y = array_api.exp(Z - broadcast_max_z)
        y = array_api.sum(y, axis=self.axes)
        y = array_api.log(y) + max_z
        return y
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_z = array_api.max(Z.realize_cached_data(), self.axes)
        if self.axes is None:
            shape = numpy.ones(len(Z.shape)).astype(int)
        else:
            shape = numpy.array(Z.shape)
            shape[list(self.axes)] = 1
        x = Z - array_api.broadcast_to(array_api.reshape(max_z, shape), Z.shape)
        y = exp(x)
        sum_y = summation(y, axes=self.axes)
        grad = out_grad / sum_y
        grad = broadcast_to(grad.reshape(shape), Z.shape)
        return grad * y
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

