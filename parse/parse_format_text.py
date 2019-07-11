'''https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto'''

from tensorflow.python.framework import tensor_util
from solver import Range
import ast
import numpy as np


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
    value = []
    print(attrs)
    while True:
        x = input("Please specify the range of inputs\n"
                  "e.g. [[-1, 1], [0, None]] means the first range is [-1, 1] and the second range is [0 ,inf):\n")
        try:
            input_list = ast.literal_eval(x)
            if isinstance(input_list, list) and np.array(input_list) == (len(dtypes), 2):
                break
        except:
            pass

    for (i, rng) in enumerate(input_list):
        if None in rng:
            value.append(Range(name="oneshotiterator", dtype=dtypes[i]))
            if rng[0] is not None:
                value[-1].left = rng[0]
            if rng[1] is not None:
                value[-1].right = rng[1]
        else:
            value.append(Range(left=rng[0], right=rng[1]))

    return value


def placeholder(attrs):
    dtype = attrs["dtype"].type
    print(attrs)
    while True:
        x = input("Please specify the range of the placeholder \n"
                  "e.g. [-1, 1] means the range is [-1, 1] \n"
                  "e,g, [0, None] means the range is [0 ,inf):\n")
        try:
            rng = ast.literal_eval(x)
            if isinstance(rng, list) and len(rng) == 2:
                break
        except:
            pass

    if None in rng:
        value = Range(name="oneshotiterator", dtype=dtype)
        if rng[0] is not None:
            value.left = rng[0]
        if rng[1] is not None:
            value.right = rng[1]
        return value
    else:
        return Range(left=rng[0], right=rng[1])
