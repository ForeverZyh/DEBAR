'''https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto'''

from tensorflow.python.framework import tensor_util
from solver import Range
import ast
import numpy as np
import z3
from utils import *

placeholder_map = {}


def const(node):
    attrs = node.attr
    tensor = attrs["value"].tensor
    value = tensor_util.MakeNdarray(tensor)
    return value


def iteratorv2(node):
    attrs = node.attr
    return oneshotiterator(node)


def variablev2(node):
    attrs = node.attr
    dtype = attrs["dtype"].type
    shape = attrs["dtype"].shape
    if dtype in [1, 2, 19] and len(shape_from_proto(shape)) > 0:
        return Range(left=-1, right=1)
    else:
        return placeholder(node)


def oneshotiterator(node):
    if node.name in placeholder_map:
        return placeholder_map[node.name]
    attrs = node.attr
    dtypes = attrs["output_types"].list.type
    value = []
    print(node)
    while True:
        x = input("Please specify the range of inputs\n"
                  "e.g. [[-1, 1], [0, None]] means the first range is [-1, 1] and the second range is [0 ,inf):\n")
        try:
            input_list = ast.literal_eval(x)
        except:
            input_list = None

        if not isinstance(input_list, list):
            print("Input string is not a list!")
        elif np.array(input_list).shape != (len(dtypes), 2):
            print("Input list's shape not match with %s (received %s)!" % (
                str((len(dtypes), 2)), str(np.array(input_list))))
        else:
            break

    for (i, rng) in enumerate(input_list):
        if None in rng:
            value.append(Range(left=rng[0] if rng[0] is not None else -OVERFLOW_LIMIT,
                               right=rng[1] if rng[1] is not None else OVERFLOW_LIMIT))
        else:
            value.append(Range(left=rng[0], right=rng[1]))

    placeholder_map[node.name] = value
    return placeholder_map[node.name]


def placeholder(node):
    if node.name in placeholder_map:
        return placeholder_map[node.name]
    attrs = node.attr
    dtype = attrs["dtype"].type
    print(node)
    while True:
        x = input("Please specify the range of the placeholder \n"
                  "e.g. [-1, 1] means the range is [-1, 1] \n"
                  "e,g, [0, None] means the range is [0 ,inf):\n")
        try:
            rng = ast.literal_eval(x)
        except:
            rng = None

        if isinstance(rng, list) and len(rng) == 2:
            break

    if None in rng:
        placeholder_map[node.name] = Range(left=rng[0] if rng[0] is not None else -OVERFLOW_LIMIT,
                                           right=rng[1] if rng[1] is not None else OVERFLOW_LIMIT)
    else:
        placeholder_map[node.name] = Range(left=rng[0], right=rng[1])

    return placeholder_map[node.name]
