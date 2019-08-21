from analysis.abstract_interpretation import AbstractInterpretation
import parse.parse_format_text as parse_format_text
import math
import copy
import warnings
from solver import Range, Solver, Array
import numpy as np
import z3
from utils import OVERFLOW_D, UNDERFLOW_D, OVERFLOW_LIMIT, UNDERFLOW_LIMIT
from utils import resolve_type
from itertools import combinations_with_replacement, product

turn_on_bool = False


def real_size(a, b):
    if str(a) == "?" and str(b) == "?":
        raise AssertionError("cannot infer ? size")
    elif str(a) == "?":
        return int(b)
    else:
        return int(a)


def dumy():
    return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)


def safeexp(X):
    UPPER_BOUND = 100
    try:
        ans = []
        for x in X:
            ans.append(min(math.exp(min(x, UPPER_BOUND)), OVERFLOW_LIMIT))
        return np.array(ans)
    except:
        return min(math.exp(min(X, UPPER_BOUND)), OVERFLOW_LIMIT)


class InferValue:
    @staticmethod
    def add(args: list, node):
        assert len(args) == 2
        if args[0].value is None or args[1].value is None:
            return None
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            x = InferValue.expanddims([args[0]], node)
            y = InferValue.expanddims([args[1]], node)
            return Range(left=x.left + y.left, right=x.right + y.right)
        else:
            return args[0].value + args[1].value

    @staticmethod
    def addn(args: list, node):
        assert len(args) > 0
        if len(args) == 1:
            return args[0].value
        else:
            s = InferValue.add([args[0], args[1]], node)
            for i in range(2, len(args)):
                s = InferValue.add([AbstractInterpretation(value=s), args[i]], node)
            return s

    @staticmethod
    def all(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), True,
        #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), False, True)),
        #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), False,
        #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), True, True)))

    @staticmethod
    def any(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), True,
        #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), False, True)),
        #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), False,
        #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), True, True)))

    @staticmethod
    def argmax(args: list, node):
        assert len(args) == 2
        try:
            return Range(left=0, right=int(args[0].size[int(args[1].value)]) - 1)
        except:
            return Range(left=0, right=OVERFLOW_LIMIT)

    @staticmethod
    def assign(args: list, node):
        assert len(args) == 2
        if args[0].value is None:
            return args[1].value
        else:
            return args[0].value

    def assignadd(args: list, node):
        y = InferValue.expanddims([args[1]], node)
        if y.left == 0:
            args[0].value.left = 0
        elif y.left < 0:
            args[0].value.left = -OVERFLOW_LIMIT
        else:
            args[0].value.left = y.left
        if y.right == 0:
            args[0].value.right = 0
        elif y.right > 0:
            args[0].value.right = OVERFLOW_LIMIT
        else:
            args[0].value.right = y.right
        return args[0].value

    @staticmethod
    def avgpool(args: list, node):
        assert len(args) == 1
        return InferValue.expanddims(args, node)

    @staticmethod
    def batchmatmul(args: list, node):
        assert len(args) == 2
        x = copy.deepcopy(args[0])
        y = copy.deepcopy(args[1])
        x.size = x.size[1:]
        y.size = y.size[1:]
        return InferValue.matmul([x, y], node)

    @staticmethod
    def biasadd(args: list, node):
        assert len(args) == 2 and len(args[1].size) == 1 and (
                str(args[0].size[-1]) == "?" or str(args[1].size[0]) or args[0].size[-1] == args[1].size[0])
        # ind = real_size(args[0].size[-1], args[1].size[0])
        return Range(left=args[0].value.left + args[1].value.left,
                     right=args[0].value.right + args[1].value.right)

    @staticmethod
    def cast(args: list, node):
        assert len(args) == 1
        attrs = node.attr
        if int(attrs['SrcT'].type) in [3] and int(attrs['DstT'].type) in [1]:
            if isinstance(args[0].value, Range):
                return args[0].value
            else:
                try:
                    return float(args[0].value)
                except:
                    return Range(left=float(np.min(args[0].value)), right=float(np.max(args[0].value)))
        elif int(attrs['SrcT'].type) in [10] and int(attrs['DstT'].type) in [1]:
            # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), 0,
            #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), 1, 0)),
            #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), 0,
            #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), 1, 1)))
            return Range(left=0, right=1)
        elif int(attrs['SrcT'].type) in [10] and int(attrs['DstT'].type) in [3]:
            # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), 0,
            #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), 1, 0)),
            #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), 0,
            #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), 1, 1)))
            return Range(left=0, right=1)
        elif int(attrs['SrcT'].type) in [1] and int(attrs['DstT'].type) in [10]:
            # return Range(left=z3.If(z3.And(args[0].value.left == 0, args[0].value.right == 0), True,
            #                         z3.If(z3.And(args[0].value.left == 0, args[0].value.right == 0), False, True)),
            #              right=z3.If(z3.And(args[0].value.left == 0, args[0].value.right == 0), False,
            #                          z3.If(z3.And(args[0].value.left == 0, args[0].value.right == 0), True, True)))
            return Range(left=True, right=True)
        elif int(attrs['SrcT'].type) in [9] and int(attrs['DstT'].type) in [3]:
            return args[0].value
        elif int(attrs['SrcT'].type) in [3] and int(attrs['DstT'].type) in [9]:
            return args[0].value
        elif int(attrs['SrcT'].type) in [1] and int(attrs['DstT'].type) in [3]:
            return InferValue.floor(args, node)
        else:
            raise NotImplementedError("%s -> %s not implemented!" % (attrs['SrcT'].type, attrs['DstT'].type))

    @staticmethod
    def checknumerics(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def clipbyvalue(args: list, node):
        assert len(args) == 3
        if isinstance(args[0].value, Range):
            return Range(left=max(args[0].value.left,
                                  float(args[1].value) if not isinstance(args[1].value, Range) else args[1].value.left),
                         right=min(args[0].value.right,
                                   float(args[2].value) if not isinstance(args[2].value, Range) else args[
                                       2].value.right))
        else:
            return min(max(args[0].value, args[1].value), args[2].value)

    @staticmethod
    def concatv2(args: list, node):
        return InferValue.pack(args[:-1], node)

    @staticmethod
    def const(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def conv2d(args: list, node):
        assert len(args) == 2
        ind = 1
        for x in args[1].size[:-1]:
            ind *= int(x)
        x = InferValue.expanddims([args[0]], node)
        y = InferValue.expanddims([args[1]], node)
        ends = [x.left * y.left * ind, x.left * y.right * ind,
                x.right * y.left * ind, x.right * y.right * ind]
        return Range(left=min(ends), right=max(ends))

    @staticmethod
    def depthwiseconv2dnative(args: list, node):
        assert len(args) == 2
        ind = 1
        for x in args[1].size[:2]:
            ind *= int(x)
        ends = [args[0].value.left * args[1].value.left * ind, args[0].value.left * args[1].value.right * ind,
                args[0].value.right * args[1].value.left * ind, args[0].value.right * args[1].value.right * ind]
        return Range(left=min(ends), right=max(ends))

    @staticmethod
    def dynamicstitch(args: list, node):
        assert len(args) % 2 == 0
        datas = args[len(args) // 2:]
        return InferValue.pack(datas, node)

    @staticmethod
    def enter(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def equal(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
        #     return np.equal(args[0].value, args[1].value)
        # else:
        #     x = InferValue.expanddims([args[0]], node)
        #     y = InferValue.expanddims([args[1]], node)
        #     condition = z3.And(x.left == y.left, x.right == y.right, x.left == x.right)
        #     return Range(left=z3.If(condition, False, True), right=z3.If(condition, True, True))

    @staticmethod
    def exit(args: list, node):
        return InferValue.identity(args, node)

    @staticmethod
    def expanddims(args: list, node):
        return args[0].value if isinstance(args[0].value, Range) else Range(left=resolve_type(np.min(args[0].value)),
                                                                            right=resolve_type(np.max(args[0].value)))

    @staticmethod
    def fill(args: list, node):
        assert len(args) == 2
        return args[1].value

    @staticmethod
    def floor(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=math.floor(args[0].value.left), right=math.floor(args[0].value.right))
        else:
            return np.floor(args[0].value)

    @staticmethod
    def fusedbatchnorm(args: list, node):
        assert len(args) == 5
        # x, scale, offset, mean, variance
        epsilon = float(node.attr['epsilon'].f)
        is_training = node.attr["is_training"].b

        x = InferValue.expanddims([args[0]], node)
        mean = InferValue.expanddims([args[1]], node)
        variance = InferValue.expanddims([args[2]], node) + epsilon

        if not is_training:
            offset = InferValue.expanddims([args[3]], node)
            scale = InferValue.expanddims([args[4]], node)
            ends_scale_variance = [scale.left / variance.left, scale.right / variance.left,
                                   scale.left / variance.right,
                                   scale.right / variance.right]

            ends = [(x.left - mean.right) * end for end in ends_scale_variance] + [
                (x.right - mean.left) * end for end in ends_scale_variance]

            return [Range(left=min(ends) + offset.left, right=max(ends) + offset.right),
                    dumy(), dumy(), dumy(), dumy()]
        else:
            ends_scale_variance = [1 / variance.left, 1 / variance.right]

            ends = [(x.left - mean.right) * end for end in ends_scale_variance] + [
                (x.right - mean.left) * end for end in ends_scale_variance]

            return [Range(left=min(ends), right=max(ends)), dumy(), dumy(), dumy(), dumy()]

    @staticmethod
    def gatherv2(args: list, node):
        assert len(args) == 3
        return InferValue.expanddims(args, node)

    @staticmethod
    def greater(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
        #     return np.greater(args[0].value, args[1].value)
        # else:
        #     x = InferValue.expanddims([args[0]], node)
        #     y = InferValue.expanddims([args[1]], node)
        #     return Range(left=z3.If(x.left > y.right, False, z3.If(x.right <= y.left, True, True)),
        #                  right=z3.If(x.left > y.right, True, z3.If(x.right <= y.left, False, True))
        #                  )

    @staticmethod
    def greaterequal(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
        #     return np.greater_equal(args[0].value, args[1].value)
        # else:
        #     x = InferValue.expanddims([args[0]], node)
        #     y = InferValue.expanddims([args[1]], node)
        #     return Range(left=z3.If(x.left >= y.right, False, z3.If(x.right < y.left, True, True)),
        #                  right=z3.If(x.left >= y.right, True, z3.If(x.right < y.left, False, True))
        #                  )

    @staticmethod
    def identity(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def iteratorgetnext(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def iteratorv2(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def less(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
        #     return np.less(args[0].value, args[1].value)
        # else:
        #     x = InferValue.expanddims([args[0]], node)
        #     y = InferValue.expanddims([args[1]], node)
        #     return Range(left=z3.If(x.left >= y.right, True, z3.If(x.right < y.left, False, True)),
        #                  right=z3.If(x.left >= y.right, False, z3.If(x.right < y.left, True, True))
        #                  )

    @staticmethod
    def linspace(args: list, node):
        assert len(args) == 3
        return np.linspace(args[0].value, args[1].value, args[2].value)

    @staticmethod
    def logicaland(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if args[0].value is None or args[1].value is None:
        #     return
        # cond1 = z3.Or(z3.And(args[0].value.left, z3.Not(args[0].value.right)),
        #               z3.And(args[1].value.left, z3.Not(args[1].value.right)))
        # cond2 = z3.And(z3.Not(args[0].value.left), z3.Not(args[1].value.left), args[0].value.right,
        #                args[1].value.right)
        # return Range(left=z3.If(cond1, True, z3.If(cond2, False, True)),
        #              right=z3.If(cond1, False, z3.If(cond2, True, True)))

    @staticmethod
    def logicalnot(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 1
        # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), False,
        #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), True, True)),
        #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), True,
        #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), False, True)))

    @staticmethod
    def logicalor(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError
        # assert len(args) == 2
        # if args[0].value is None or args[1].value is None:
        #     return
        # cond1 = z3.Or(z3.And(args[0].value.right, z3.Not(args[0].value.left)),
        #               z3.And(args[1].value.right, z3.Not(args[1].value.left)))
        # cond2 = z3.And(z3.Not(args[0].value.right), z3.Not(args[1].value.right), args[0].value.left,
        #                args[1].value.left)
        # return Range(left=z3.If(cond1, False, z3.If(cond2, True, True)),
        #              right=z3.If(cond1, True, z3.If(cond2, False, True)))

    @staticmethod
    def loguniformcandidatesampler(args: list, node):
        assert len(args) == 1
        ind = int(node.attr["range_max"].i)
        num = int(node.attr["num_sampled"].i)
        return [Range(left=0, right=ind - 1), Range(left=UNDERFLOW_LIMIT * 10, right=num),
                Range(left=UNDERFLOW_LIMIT * 10, right=num)]

    @staticmethod
    def loopcond(args: list, node):
        return InferValue.identity(args, node)

    @staticmethod
    def matmul(args: list, node):
        assert len(args) == 2 and len(args[0].size) == len(args[1].size)
        for i in range(len(args[0].size) - 2):
            assert str(args[0].size[i]) == "?" or str(args[1].size[i]) == "?" or args[0].size[i] == args[1].size[i]
        ind = real_size(args[0].size[-1], args[1].size[-2])
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            return np.matmul(args[0].value, args[1].value)
        else:
            x = InferValue.expanddims([args[0]], node)
            y = InferValue.expanddims([args[1]], node)
            ends = [x.left * y.left * ind, x.left * y.right * ind, x.right * y.left * ind, x.right * y.right * ind]
            return Range(left=min(ends), right=max(ends))

    @staticmethod
    def max(args: list, node):
        assert len(args) == 2
        x = args[0].value
        y = args[1].value
        if isinstance(x, Range) and isinstance(y, Range):
            return Range(left=max(x.left, y.left), right=max(x.right, y.right))
        elif not isinstance(x, Range) and not isinstance(y, Range):
            return np.maximum(x, y)
        else:
            if isinstance(y, Range):
                x, y = y, x
            y = resolve_type(np.max(y))
            return Range(left=max(x.left, y), right=max(x.right, y))

    @staticmethod
    def maxpool(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def maximum(args: list, node):
        return InferValue.max(args, node)

    @staticmethod
    def mean(args: list, node):
        assert len(args) == 2
        return InferValue.expanddims(args, node)

    @staticmethod
    def merge(args: list, node):
        tmp = InferValue.pack(args, node)
        max_index = int(node.attr["N"].i)
        return_index = Range(left=0, right=max_index - 1)
        if isinstance(tmp, tuple):
            raise AssertionError
        else:
            return [tmp, return_index]

    @staticmethod
    def min(args: list, node):
        assert len(args) == 2
        x = args[0].value
        y = args[1].value
        if isinstance(x, Range) and isinstance(y, Range):
            return Range(left=min(x.left, y.left), right=min(x.right, y.right))
        elif not isinstance(x, Range) and not isinstance(y, Range):
            return np.minimum(x, y)
        else:
            if isinstance(y, Range):
                x, y = y, x
            y = resolve_type(np.min(y))
            return Range(left=min(x.left, y), right=min(x.right, y))

    @staticmethod
    def minimum(args: list, node):
        return InferValue.min(args, node)

    @staticmethod
    def mul(args: list, node):
        assert len(args) == 2
        if args[0].value is None or args[1].value is None:
            return None
        if isinstance(args[1].value, Range) or isinstance(args[0].value, Range):
            x = InferValue.expanddims([args[0]], node)
            y = InferValue.expanddims([args[1]], node)
            ends = [x.left * y.left, x.left * y.right, x.right * y.left, x.right * y.right]
            return Range(left=min(ends), right=max(ends))
        else:
            return args[0].value * args[1].value

    @staticmethod
    def neg(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=-args[0].value.right, right=-args[0].value.left)
        else:
            return -args[0].value

    @staticmethod
    def nonmaxsuppressionv3(args: list, node):
        assert len(args) == 5
        try:
            ind = int(args[1].size[0])
            return Range(left=0, right=ind - 1)
        except:
            return Range(left=0, right=OVERFLOW_LIMIT)

    @staticmethod
    def notequal(args: list, node):
        if not turn_on_bool:
            return Range(left=True, right=True)
        raise NotImplementedError

    @staticmethod
    def onehot(args: list, node):
        assert len(args) == 4
        return Range(left=min([args[2].value, args[3].value]),
                     right=max([args[2].value, args[3].value]))

    @staticmethod
    def oneshotiterator(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def pack(args: list, node):
        try:
            maxs = [args[i].value.right if isinstance(args[i].value, Range) else resolve_type(np.max(args[i].value)) for
                    i in range(len(args))]
            mins = [args[i].value.left if isinstance(args[i].value, Range) else resolve_type(np.min(args[i].value)) for
                    i in range(len(args))]
            if None in maxs or None in mins:
                return None
            return Range(left=min(mins), right=max(maxs))
        except:
            # boolean
            # has_zero = []
            # has_one = []
            # for i in range(len(args)):
            #     if isinstance(args[i].value, Range):
            #         has_zero.append(args[i].value.left)
            #         has_one.append(args[i].value.right)
            #     else:
            #         has_zero.append(not bool(np.all(args[i].value)))
            #         has_one.append(bool(np.any(args[i].value)))
            # return Range(left=z3.Or(has_zero), right=z3.Or(has_one))
            return Range(left=True, right=True)

    @staticmethod
    def pad(args: list, node):
        return InferValue.expanddims(args, node)

    @staticmethod
    def paddingfifoqueuev2(args: list, node):
        return InferValue.randomshufflequeuev2(args, node)

    @staticmethod
    def placeholder(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def prod(args: list, node):
        assert len(args) == 2
        return InferValue.mul(args, node)

    @staticmethod
    def queuedequeuemanyv2(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def randomshuffle(args: list, node):
        assert len(args) == 1
        return InferValue.expanddims(args, node)

    @staticmethod
    def randomshufflequeuev2(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, "placeholder")(node)

    @staticmethod
    def randomstandardnormal(args: list, node):
        assert len(args) == 1
        return Range(left=UNDERFLOW_LIMIT * 10, right=1)

    @staticmethod
    def randomuniform(args: list, node):
        assert len(args) == 1
        return Range(left=UNDERFLOW_LIMIT * 10, right=1)

    @staticmethod
    def range(args: list, node):
        assert len(args) == 3
        left = args[0].value.left if isinstance(args[0].value, Range) else int(args[0].value)
        right = args[1].value.right if isinstance(args[1].value, Range) else int(args[1].value)
        return Range(left=left, right=right)

    @staticmethod
    def rank(args: list, node):
        assert len(args) == 1
        try:
            return int(args[0].size)
        except:
            return Range(left=1, right=OVERFLOW_LIMIT)

    @staticmethod
    def readvariableop(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def realdiv(args: list, node):
        assert len(args) == 2
        try:
            x = float(args[0].value)
        except:
            x = InferValue.expanddims([args[0]], node)
        try:
            y = float(args[1].value)
        except:
            y = InferValue.expanddims([args[1]], node)

        if isinstance(x, Range) and isinstance(y, Range):
            if y.left > 0 or y.right < 0:
                ends = [x.left / y.left, x.left / y.right, x.right / y.left, x.right / y.right]
                return Range(left=min(ends), right=max(ends))
            else:
                return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)
        elif not isinstance(y, Range):
            return x * (1 / y)
        else:
            if y.left > 0 or y.right < 0:
                ends = [x / y.left, x / y.right]
                return Range(left=min(ends), right=max(ends))
            else:
                return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)

    @staticmethod
    def relu(args: list, node):
        assert len(args) == 1
        return Range(left=max([args[0].value.left, 0]),
                     right=max([args[0].value.right, 0]))

    @staticmethod
    def relu6(args: list, node):
        assert len(args) == 1
        return Range(left=min(max(args[0].value.left, 0), 6),
                     right=min(max(args[0].value.right, 0), 6))

    @staticmethod
    def reshape(args: list, node):
        assert len(args) == 2
        return InferValue.expanddims(args, node)

    @staticmethod
    def resizebilinear(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def reversev2(args: list, node):
        assert len(args) == 2
        return InferValue.expanddims(args, node)

    @staticmethod
    def rsqrt(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left = math.sqrt(args[0].value.left)
            right = math.sqrt(args[0].value.right)
            return Range(left=1 / right, right=1 / left)
        else:
            return 1 / math.sqrt(args[0].value)

    @staticmethod
    def select(args: list, node):
        assert len(args) == 3
        if not isinstance(args[0].value, Range):
            # print(args[0].value)
            raise NotImplementedError("not implemented when the condition is known")
        x = InferValue.expanddims([args[1]], node)
        y = InferValue.expanddims([args[2]], node)
        if not turn_on_bool:
            return Range(left=min(x.left, y.left), right=max(x.right, y.right))
        raise NotImplementedError
        # return Range(left=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), x.left,
        #                         z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), y.left,
        #                               z3.If(x.left < y.left, x.left, y.left))),
        #              right=z3.If(z3.And(args[0].value.left, z3.Not(args[0].value.right)), x.right,
        #                          z3.If(z3.And(args[0].value.right, z3.Not(args[0].value.left)), y.right,
        #                                z3.If(x.right > y.right, x.right, y.right))))

    @staticmethod
    def shape(args: list, node):
        assert len(args) == 1
        try:
            return [int(x) for x in args[0].size]
        except:
            return Range(left=1, right=OVERFLOW_LIMIT)

    @staticmethod
    def size(args: list, node):
        assert len(args) == 1
        try:
            ele = 1
            for x in args[0].size:
                ele *= int(x)
            if ele < 0:
                return Range(left=0, right=OVERFLOW_LIMIT)
            else:
                return ele
        except:
            return Range(left=0, right=OVERFLOW_LIMIT)

    @staticmethod
    def slice(args: list, node):
        return InferValue.expanddims(args, node)

    @staticmethod
    def split(args: list, node):
        assert len(args) == 2
        nums = int(node.attr["num_split"].i)
        if nums == 1:
            return InferValue.expanddims(args[1:], node)
        else:
            return [InferValue.expanddims(args[1:], node) for _ in range(nums)]

    @staticmethod
    def sqrt(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left = math.sqrt(args[0].value.left)
            right = math.sqrt(args[0].value.right)

            return Range(left=left, right=right)
        else:
            return np.sqrt(args[0].value)

    @staticmethod
    def square(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left_sq = args[0].value.left * args[0].value.left
            right_sq = args[0].value.right * args[0].value.right
            min_sq = min(left_sq, right_sq)
            max_sq = max(left_sq, right_sq)
            cond = args[0].value.left <= 0 and args[0].value.right >= 0
            return Range(left=0 if cond else min_sq, right=max_sq)
        else:
            return args[0].value * args[0].value

    @staticmethod
    def squareddifference(args: list, node):
        assert len(args) == 2
        value1 = (args[0].value.left - args[1].value.right) * (args[0].value.left - args[1].value.right)
        value2 = (args[0].value.right - args[1].value.left) * (args[0].value.right - args[1].value.left)
        return InferValue.square([AbstractInterpretation(value=Range(left=value1, right=value2))], node)

    @staticmethod
    def squeeze(args: list, node):
        assert len(args) == 1
        return InferValue.expanddims(args, node)

    @staticmethod
    def stopgradient(args: list, node):
        return InferValue.identity(args, node)

    @staticmethod
    def stridedslice(args: list, node):
        return InferValue.expanddims(args, node)

    @staticmethod
    def sub(args: list, node):
        assert len(args) == 2
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            x = InferValue.expanddims([args[0]], node)
            y = InferValue.expanddims([args[1]], node)
            return Range(left=x.left - y.right, right=x.right - y.left)
        else:
            return args[0].value - args[1].value

    @staticmethod
    def sum(args: list, node):
        assert len(args) == 2
        if args[0].value is None:
            return None
        if isinstance(args[0].value, Range):
            try:
                ind = int(args[0].size[int(args[1].value)])
                return Range(left=args[0].value.left * ind, right=args[0].value.right * ind)
            except:
                ind = Range(left=0, right=OVERFLOW_LIMIT)
                t = InferValue.mul([args[0], AbstractInterpretation(value=ind, dtype=3, size=[])], node)
                if isinstance(t, tuple):
                    raise AssertionError
                else:
                    return t
        else:
            return np.sum(args[0].value, axis=args[1].value)

    @staticmethod
    def switch(args: list, node):
        assert len(args) == 2
        return [args[0].value, args[0].value]

    @staticmethod
    def tensorarraygatherv3(args: list, node):
        assert len(args) == 3
        return args[0].value

    @staticmethod
    def tensorarrayv3(args: list, node):
        assert len(args) == 1
        return [dumy(), dumy()]

    @staticmethod
    def tensorarrayreadv3(args: list, node):
        assert len(args) == 3
        return args[0].value

    @staticmethod
    def tensorarrayscatterv3(args: list, node):
        assert len(args) == 4
        # TODO check if dumy() works here
        if isinstance(args[2].value, Range):
            return args[0].value
        else:
            return args[0].value

    @staticmethod
    def tensorarraysizev3(args: list, node):
        assert len(args) == 2
        return int(args[0].size[0])

    @staticmethod
    def tensorarraywritev3(args: list, node):
        assert len(args) == 4
        return InferValue.tensorarrayscatterv3(args, node)

    @staticmethod
    def tile(args: list, node):
        assert len(args) == 2
        return InferValue.expanddims(args, node)

    @staticmethod
    def topkv2(args: list, node):
        assert len(args) == 2
        try:
            ind = int(args[0].size[-1])
            value = Range(left=0, right=ind - 1)
        except:
            value = Range(left=0, right=OVERFLOW_LIMIT)
        return [InferValue.expanddims(args, node), value]

    @staticmethod
    def transpose(args: list, node):
        assert len(args) == 2
        return InferValue.expanddims(args, node)

    @staticmethod
    def unpack(args: list, node):
        assert len(args) == 1
        nums = int(node.attr["num"].i)
        if nums == 1:
            return InferValue.expanddims(args, node)
        else:
            return [InferValue.expanddims(args, node) for _ in range(nums)]

    @staticmethod
    def varhandleop(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, "variablev2")(node)

    @staticmethod
    def variable(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, "variablev2")(node)

    @staticmethod
    def variablev2(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def where(args: list, node):
        assert len(args) == 1
        try:
            x = np.max(args[0].size)
            return Range(left=0, right=x - 1)
        except:
            return Range(left=0, right=OVERFLOW_LIMIT - 1)

    @staticmethod
    def zeroslike(args: list, node):
        assert len(args) == 1
        return Range(left=0, right=0)

    @staticmethod
    def floormod(args: list, node):
        warnings.warn("floormod not implemented", RuntimeWarning)

    @staticmethod
    def iteratortostringhandle(args: list, node):
        warnings.warn("iteratortostringhandle not implemented", RuntimeWarning)

    @staticmethod
    def noop(args: list, node):
        warnings.warn("noop not implemented", RuntimeWarning)

    @staticmethod
    def restorev2(args: list, node):
        warnings.warn("restorev2 not implemented", RuntimeWarning)

    @staticmethod
    def savev2(args: list, node):
        warnings.warn("savev2 not implemented", RuntimeWarning)

    # non linear operations:
    @staticmethod
    def log(args: list, node):
        assert len(args) == 1
        # stride = 15
        # intervals = [(0, math.pow(10, UNDERFLOW_D))] + \
        #             [(math.pow(10, i), math.pow(10, i + stride)) for i in range(UNDERFLOW_D, OVERFLOW_D, stride)] + \
        #             [(math.pow(10, OVERFLOW_D), math.inf)]
        if isinstance(args[0].value, Range):
            if args[0].value.left <= 0:
                return Range(left=-OVERFLOW_LIMIT, right=math.log(args[0].value.right))
            else:
                return Range(left=math.log(args[0].value.left), right=math.log(args[0].value.right))
            # value = Range(name="log", dtype=1)
            # constraints = []
            # for (left, right) in combinations_with_replacement(intervals, 2):
            #     temp_constraint = [Solver.in_interval(args[0].value.left, left),
            #                        Solver.in_interval(args[0].value.right, right)]
            #     if left[0] != 0:
            #         temp_constraint += [value.left == math.log(left[0])]
            #     if not math.isinf(right[1]):
            #         temp_constraint += [value.right == math.log(right[1])]
            #     constraints.append(z3.And(temp_constraint))
            #
            # return value, z3.And(z3.Or(constraints), value.left <= value.right)
        else:
            return math.log(args[0].value)

    @staticmethod
    def exp(args: list, node):
        assert len(args) == 1
        # intervals = [(-math.inf, -90), (-90, -5), (-5, 5), (5, 90), (90, math.inf)]
        if isinstance(args[0].value, Range):
            return Range(left=safeexp(args[0].value.left), right=safeexp(args[0].value.right))
            # value = Range(name="exp", dtype=1)
            # constraints = []
            # for (left, right) in combinations_with_replacement(intervals, 2):
            #     temp_constraint = [Solver.in_interval(args[0].value.left, left),
            #                        Solver.in_interval(args[0].value.right, right)]
            #     if not math.isinf(np.min(left)):
            #         temp_constraint += [value.left == math.exp(np.min(left))]
            #     else:
            #         temp_constraint += [value.left == 0]
            #     if not math.isinf(np.max(right)):
            #         temp_constraint += [value.right == math.exp(np.max(right))]
            #     constraints.append(z3.And(temp_constraint))
            #
            # return value, z3.And(z3.Or(constraints), value.left <= value.right)
        else:
            return math.exp(args[0].value)

    @staticmethod
    def softmax(args: list, node):
        assert len(args) == 1
        ind = int(args[0].size[-1])
        assert ind > 1
        # intervals = [(-math.inf, -90), (-90, -1), (-1, 1), (1, 90), (90, math.inf)]
        if isinstance(args[0].value, Range):
            min_ele = safeexp(args[0].value.left)
            max_ele = safeexp(args[0].value.right)
            if max_ele >= OVERFLOW_LIMIT or min_ele == 0:
                left = 0
            else:
                left = min_ele / ((ind - 1) * max_ele + min_ele)
            if max_ele >= OVERFLOW_LIMIT or min_ele == 0:
                right = 1
            else:
                right = max_ele / ((ind - 1) * min_ele + max_ele)
            return Range(left=left, right=right)
        else:
            tmp_exp = np.exp(args[0].value)
            return tmp_exp / np.sum(tmp_exp)

    @staticmethod
    def sigmoid(args: list, node):
        assert len(args) == 1
        # intervals = [(-math.inf, -40), (-40, -2), (-2, 2), (2, 40), (40, math.inf)]
        if isinstance(args[0].value, Range):
            return Range(left=1 / (1 + safeexp(-args[0].value.left)), right=1 / (1 + safeexp(-args[0].value.right)))
            # value = Range(name="sigmoid", dtype=1)
            # pre_left = None
            # pre_right = None
            # for interval in intervals:
            #     if not math.isinf(np.min(interval)):
            #         left = 1 / (1 + math.exp(-np.min(interval)))
            #     else:
            #         left = 0
            #     if not math.isinf(np.max(interval)):
            #         right = 1 / (1 + math.exp(-np.max(interval)))
            #     else:
            #         right = 1
            #
            #     if pre_left is None:
            #         pre_left = left
            #     else:
            #         pre_left = z3.If(Solver.in_interval(args[0].value.left, interval), left, pre_left)
            #     if pre_right is None:
            #         pre_right = right
            #     else:
            #         pre_right = z3.If(Solver.in_interval(args[0].value.right, interval), right, pre_right)
            #
            # return value, z3.And(value.left == pre_left, value.right == pre_right, value.left <= value.right)
        else:
            return 1 / (1 + safeexp(-args[0].value))

    @staticmethod
    def tanh(args: list, node):
        assert len(args) == 1
        # intervals = [(-math.inf, -20), (-20, -1), (-1, 1), (1, 20), (20, math.inf)]
        if isinstance(args[0].value, Range):
            return Range(left=np.tanh(args[0].value.left), right=np.tanh(args[0].value.right))
            # value = Range(name="tanh", dtype=1)
            # pre_left = None
            # pre_right = None
            # for interval in intervals:
            #     if not math.isinf(np.min(interval)):
            #         left = math.tanh(np.min(interval))
            #     else:
            #         left = -1
            #     if not math.isinf(np.max(interval)):
            #         right = math.tanh(np.max(interval))
            #     else:
            #         right = 1
            #
            #     if pre_left is None:
            #         pre_left = left
            #     else:
            #         pre_left = z3.If(Solver.in_interval(args[0].value.left, interval), left, pre_left)
            #     if pre_right is None:
            #         pre_right = right
            #     else:
            #         pre_right = z3.If(Solver.in_interval(args[0].value.right, interval), right, pre_right)
            #
            # return value, z3.And(value.left == pre_left, value.right == pre_right, value.left <= value.right)
        else:
            return np.tanh(args[0].value)


class InferArray:
    @staticmethod
    def add(args: list, node):
        # if len(args[1].size) == 0:
        #     t = float(args[1].value)
        #     ret = copy.deepcopy(args[0].array)
        #     for x in ret.block_to_symbol:
        #         ret.block_to_symbol[x] += t
        #
        #     return ret
        # else:
        try:
            len(args[0].size) == len(args[1].size)
        except:
            return None
        assert len(args) == 2 and len(args[0].size) == len(args[1].size)
        ind = len(args[0].size)
        for i in range(ind):
            try:
                l1 = int(args[0].size[i])
            except:
                l1 = -1
            try:
                l2 = int(args[1].size[i])
            except:
                l2 = -1
            assert l1 == l2

        ret = Array("tmp", args[0].size)
        ret.block_to_symbol = dict()
        ret.index_slices = Array.join_index_slices(args[0].array.index_slices, args[1].array.index_slices)
        keys0 = args[0].array.get_corresponding_keys(ret.index_slices)
        keys1 = args[1].array.get_corresponding_keys(ret.index_slices)
        i = 0
        for indexes in product(*ret.index_slices):
            ret.block_to_symbol[tuple(indexes)] = keys0[i] + keys1[i]
            i += 1

        return ret

    def sub(args: list, node):
        # if len(args[1].size) == 0:
        #     t = float(args[1].value)
        #     ret = copy.deepcopy(args[0].array)
        #     for x in ret.block_to_symbol:
        #         ret.block_to_symbol[x] -= t
        #
        #     return ret
        # else:
        try:
            len(args[0].size) == len(args[1].size)
        except:
            return None
        assert len(args) == 2 and len(args[0].size) == len(args[1].size)
        ind = len(args[0].size)
        for i in range(ind):
            try:
                l1 = int(args[0].size[i])
            except:
                l1 = -1
            try:
                l2 = int(args[1].size[i])
            except:
                l2 = -1
            assert l1 == l2

        ret = Array("tmp", args[0].size)
        ret.block_to_symbol = dict()
        ret.index_slices = Array.join_index_slices(args[0].array.index_slices, args[1].array.index_slices)
        keys0 = args[0].array.get_corresponding_keys(ret.index_slices)
        keys1 = args[1].array.get_corresponding_keys(ret.index_slices)
        i = 0
        for indexes in product(*ret.index_slices):
            ret.block_to_symbol[tuple(indexes)] = keys0[i] - keys1[i]
            i += 1

        return ret

    @staticmethod
    def concatv2(args: list, node):
        assert len(args) > 1
        if len(args) - 1 > 10:
            return None
        concat_ind = int(args[-1].value)
        for i in range(1, len(args) - 1):
            assert len(args[0].size) == len(args[i].size)
            for j in range(len(args[i].size)):
                try:
                    int(args[0].size[j])
                    int(args[i].size[j])
                except:
                    return None
                if j != concat_ind:
                    assert int(args[0].size[j]) == int(args[i].size[j])

        ret = Array("tmp", args[0].size)
        ret.block_to_symbol = dict()
        index_slices = []
        for arg in args[:-1]:
            index_slices.append(copy.deepcopy(arg.array.index_slices))
            index_slices[-1][concat_ind] = [None]

        ret.index_slices = index_slices[0]
        for i in range(1, len(args) - 1):
            ret.index_slices = Array.join_index_slices(ret.index_slices, index_slices[i])
        tmp_ret_index_slices = copy.deepcopy(ret.index_slices)
        ret.index_slices[concat_ind] = []
        split_point = 0
        for i in range(len(args) - 1):
            tmp_ret_index_slices[concat_ind] = args[i].array.index_slices[concat_ind]
            ret.index_slices[concat_ind] += list(np.array(args[i].array.index_slices[concat_ind]) + split_point)
            tmp_keys = args[i].array.get_corresponding_keys(tmp_ret_index_slices)
            tmp_ret_index_slices[concat_ind] = list(np.array(args[i].array.index_slices[concat_ind]) + split_point)
            split_point += int(args[i].array.index_slices[concat_ind][-1])
            ii = 0
            for indexes in product(*tmp_ret_index_slices):
                ret.block_to_symbol[tuple(indexes)] = tmp_keys[ii]
                ii += 1

        return ret

    @staticmethod
    def identity(args: list, node):
        assert len(args) == 1
        return args[0].array

    @staticmethod
    def zeroslike(args: list, node):
        assert len(args) == 1
        ret = Array("tmp", args[0].size)
        x = list(ret.block_to_symbol.keys())[0]
        ret.block_to_symbol[x].value = {}
        ret.block_to_symbol[x].map_to_index = {}

        return ret

    @staticmethod
    def relu(args: list, node):
        assert len(args) == 1
        ret = copy.deepcopy(args[0].array)
        ret.block_to_symbol = {}
        for x in args[0].array.block_to_symbol:
            ret.block_to_symbol[x] = args[0].array.block_to_symbol[x].relu()
        return ret

    @staticmethod
    def maximum(args: list, node):
        try:
            len(args[0].size) == len(args[1].size)
        except:
            return None
        assert len(args) == 2 and len(args[0].size) == len(args[1].size)
        one_value = list(args[1].array.block_to_symbol.values())
        if len(one_value) == 1 and len(one_value[0].value) == 0:
            return InferArray.relu([args[0]], node)
        one_value = list(args[0].array.block_to_symbol.values())
        if len(one_value) == 1 and len(one_value[0].value) == 0:
            return InferArray.relu([args[1]], node)

    @staticmethod
    def neg(args: list, node):
        assert len(args) == 1
        ret = copy.deepcopy(args[0].array)
        for x in ret.block_to_symbol:
            ret.block_to_symbol[x].neg()

        return ret

    #
    # @staticmethod
    # def exp(args: list, node):
    #     assert len(args) == 1
    #     ret = copy.deepcopy(args[0].array)
    #     constraints = []
    #     for x in ret.block_to_symbol:
    #         pre = ret.block_to_symbol[x]
    #         now = Solver.add_variable("exp", 1)
    #         ret.block_to_symbol[x] = now
    #         constraints.append(z3.Or(z3.And(pre >= 0, now >= 1), z3.And(pre < 0, now < 1, now >= 0)))
    #
    #     return ret, z3.And(constraints)

    @staticmethod
    def pack(args: list, node):
        # return InferArray.concatv2(args + [AbstractInterpretation(value=len(args[0].size) - 1)], node)
        assert len(args) >= 1
        if len(args) > 10:
            return None
        pack_ind = int(node.attr["axis"].i)
        for i in range(1, len(args)):
            try:
                len(args[0].size) == len(args[i].size)
            except:
                return None
            assert len(args[0].size) == len(args[i].size)
            for j in range(len(args[i].size)):
                try:
                    int(args[0].size[j])
                    int(args[i].size[j])
                except:
                    return None
                assert int(args[0].size[j]) == int(args[i].size[j])

        ret = Array("tmp", args[0].size)
        ret.block_to_symbol = dict()
        index_slices = []
        for arg in args:
            index_slices.append(copy.deepcopy(arg.array.index_slices))
        ret.index_slices = index_slices[0]
        for i in range(1, len(args)):
            ret.index_slices = Array.join_index_slices(ret.index_slices, index_slices[i])
        tmp_ret_index_slices = copy.deepcopy(ret.index_slices)
        ret.index_slices = ret.index_slices[:pack_ind] + [[]] + ret.index_slices[pack_ind:]

        for i in range(len(args)):
            ret.index_slices[pack_ind] += [i + 1]
            tmp_keys = args[i].array.get_corresponding_keys(tmp_ret_index_slices)
            ii = 0
            for indexes in product(*tmp_ret_index_slices):
                tmp_key = list(indexes)
                tmp_key = tmp_key[:pack_ind] + [i + 1] + tmp_key[pack_ind:]
                ret.block_to_symbol[tuple(tmp_key)] = tmp_keys[ii].add_pack_ind(pack_ind)
                ii += 1

        return ret

    @staticmethod
    def transpose(args: list, node):
        assert len(args) == 2
        assert not isinstance(args[1].value, Range)
        ret = Array("tmp", args[0].size)
        ret.index_slices = []
        ret.block_to_symbol = {}
        perm = np.array(args[1].value)
        for x in perm:
            ret.index_slices.append(args[0].array.index_slices[x])
        for indexes in product(*args[0].array.index_slices):
            new_indexes = ()
            for x in perm:
                new_indexes += (indexes[x],)

            ret.block_to_symbol[new_indexes] = args[0].array.block_to_symbol[tuple(indexes)].transpose(perm)

        return ret

    @staticmethod
    def unpack(args: list, node):
        assert len(args) == 1
        axis = int(node.attr["axis"].i)
        index_slices = copy.deepcopy(args[0].array.index_slices)
        try:
            if int(args[0].size[axis]) > 10:
                return None
        except:
            return None

        rets = []
        for i in range(int(args[0].size[axis])):
            rets.append(Array("tmp", args[0].size))
            rets[-1].index_slices = index_slices[:axis] + index_slices[axis + 1:]
            rets[-1].block_to_symbol = {}

        length = index_slices[axis][-1]
        index_slices[axis] = list(range(1, length + 1))  # e.g., 4 -> [1,2,3,4]
        tmp_keys = args[0].array.get_corresponding_keys(index_slices)
        ii = 0
        for indexes in product(*index_slices):
            tmp_key = list(indexes)
            which = indexes[axis] - 1
            tmp_key = tmp_key[:axis] + tmp_key[axis + 1:]
            rets[which].block_to_symbol[tuple(tmp_key)] = tmp_keys[ii].remove_unpack_axis(axis)
            ii += 1

        return rets if len(rets) > 1 else rets[0]

    # @staticmethod
    # def split(args: list, node):
    #     assert len(args) == 2
    #     axis = int(args[0].value)
    #     rets = []
    #     index_slices = copy.deepcopy(args[1].array.index_slices)
    #     try:
    #         if int(args[1].size[axis]) > 10:
    #             return None
    #     except:
    #         return None
    #
    #     for i in range(int(args[1].size[axis])):
    #         rets.append(Array("tmp", args[1].size))
    #         rets[-1].index_slices = copy.deepcopy(args[1].array.index_slices)
    #         rets[-1].index_slices[axis] = [1]
    #         rets[-1].block_to_symbol = {}
    #
    #         index_slices[axis] = [i]
    #         tmp_keys = args[1].array.get_corresponding_keys(index_slices)
    #         ii = 0
    #         for indexes in product(*index_slices):
    #             tmp_key = list(indexes)
    #             tmp_key[axis] = 1
    #             rets[-1].block_to_symbol[tuple(tmp_key)] = tmp_keys[ii]
    #             ii += 1
    #
    #     return rets if len(rets) > 1 else rets[0]
