'''https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto'''

from tensorflow.python.framework import tensor_util
from solver import Range
import z3


def const(attrs):
    tensor = attrs["value"].tensor
    value = tensor_util.MakeNdarray(tensor)
    return value


def iteratorv2(attrs):
    dtypes = attrs["output_types"].list.type
    # print(dtypes)
    value = [Range(name="iteratorv2", dtype=dtype) for dtype in dtypes]
    return value

def variablev2(attrs):
    dtype = attrs["dtype"].type
    # value = Range(name="variablev2", dtype=dtype)
    # return value, value.left <= value.right
    # return value, z3.And([value.left <= value.right, value.left >= -1, value.right <= 1])
    # return value, z3.And([value.left == -1, value.right == 1])
    if dtype in [1]:
        return Range(left=-1, right=1)
    else:
        return Range(name="variablev2", dtype=dtype)


def oneshotiterator(attrs):
    dtypes = attrs["output_types"].list.type
    value = [Range(name="oneshotiterator", dtype=dtype) for dtype in dtypes]
    """hard code assign for MNIST dataset"""
    value[0].left = -1
    value[0].right = 1
    value[1].left = 0
    value[1].right = 9
    return value  # , z3.And([v.left <= v.right for v in value[:1]])


def placeholder(attrs):
    dtype = attrs["dtype"].type
    """hard code assign for MNIST dataset"""
    if dtype == 3:
        return Range(left=0, right=9)
    else:
        if str(attrs["shape"].shape) == "":
            # keep_prob
            return Range(left=0.1, right=1)
        else:
            # image
            return Range(left=-1, right=1)
