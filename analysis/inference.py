import math
import copy
import warnings
import numpy as np
from itertools import product

from analysis.abstract_interpretation import AbstractInterpretation
import parse.parse_format_text as parse_format_text
from solver import Range, Array
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT, resolve_type

turn_on_bool = False
length_unknown = 1e3


# infer size from a and b under assumption that a = b, even though one of them might be unknown, i.e., equals to ?
def real_size(a, b):
    if str(a) == "?" and str(b) == "?":
        raise AssertionError("cannot infer ? size")
    elif str(a) == "?":
        return int(b)
    else:
        return int(a)


# the abstract interpretation of identity.
def identity(args, node=None):
    try:
        return args[0].value if isinstance(args[0].value, Range) else Range(left=resolve_type(np.min(args[0].value)),
                                                                            right=resolve_type(np.max(args[0].value)))
    except:  # if it is not able to get the range (e.g., it is a zero-size array)
        return None


# the abstract interpretation of joining of a list of interval abstractions.
def packtorange(args, node):
    maxs = []
    mins = []
    for arg in args:
        if isinstance(arg.value, Range):
            maxs.append(arg.value.right)
            mins.append(arg.value.left)
        elif arg.value.size > 0:
            maxs.append(resolve_type(np.max(arg.value)))
            mins.append(resolve_type(np.min(arg.value)))

    if None in maxs or None in mins:
        return None
    return Range(left=np.min(mins), right=np.max(maxs))


# returns an unbounded interval abstraction with [-inf, +inf]
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


def safesqrt(X):
    try:
        ans = []
        for x in X:
            if x < 0:
                ans.append(0)
            else:
                ans.append(math.sqrt(x))
        return np.array(ans)
    except:
        if X < 0:
            return 0
        else:
            return math.sqrt(X)


def safepow(X, Y):
    UPPER_BOUND = 100
    try:
        ans = []
        for (x, y) in zip(X, Y):
            try:
                ans.append(min(math.pow(x, y), OVERFLOW_LIMIT))
            except:
                ans.append(OVERFLOW_LIMIT)
        return np.array(ans)
    except:
        try:
            return min(math.pow(X, Y), OVERFLOW_LIMIT)
        except:
            return OVERFLOW_LIMIT


def safelgamma(X):
    try:
        ans = []
        for x in X:
            if x <= UNDERFLOW_LIMIT:
                ans.append(OVERFLOW_LIMIT)
            else:
                ans.append(math.lgamma(x))
        return np.array(ans)
    except:
        if X <= UNDERFLOW_LIMIT:
            return OVERFLOW_LIMIT
        else:
            return math.lgamma(X)


def safesoftplus(X):
    UPPER_BOUND = 100
    try:
        ans = []
        for x in X:
            if X > UPPER_BOUND:
                ans.append(X)
            else:
                ans.append(np.log1p(np.exp(X)))
        return np.array(ans)
    except:
        if X > UPPER_BOUND:
            return X
        else:
            return np.log1p(np.exp(X))


# contains the abstract interpretations of TensorFlow APIs used in interval abstraction + tensor smashing.
class InferValue:
    @staticmethod
    def abs(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left_sq = np.abs(args[0].value.left)
            right_sq = np.abs(args[0].value.right)
            min_sq = min(left_sq, right_sq)
            max_sq = max(left_sq, right_sq)
            cond = args[0].value.left <= 0 and args[0].value.right >= 0
            return Range(left=0 if cond else min_sq, right=max_sq)
        else:
            return np.abs(args[0].value)

    @staticmethod
    def add(args: list, node):
        assert len(args) == 2
        if args[0].value is None or args[1].value is None:
            return None
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            x = identity([args[0]], node)
            y = identity([args[1]], node)
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
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def any(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def argmax(args: list, node):
        assert len(args) == 2
        try:
            return Range(left=0, right=int(args[0].size[int(args[1].value)]) - 1)
        except:
            return Range(left=0, right=length_unknown)

    @staticmethod
    def assign(args: list, node):
        assert len(args) == 2
        if args[0].value is None:
            return args[1].value
        else:
            return args[0].value

    def assignadd(args: list, node):
        y = identity([args[1]], node)
        tmp = dumy()
        if y.left >= 0:
            tmp.left = args[0].value.left
        if y.right <= 0:
            tmp.right = args[0].value.right
        return tmp

    def assignsub(args: list, node):
        y = identity([args[1]], node)
        tmp = dumy()
        if y.left <= 0:
            tmp.left = args[0].value.left
        if y.right >= 0:
            tmp.right = args[0].value.right
        return tmp

    @staticmethod
    def avgpool(args: list, node):
        assert len(args) == 1
        return identity(args, node)

    @staticmethod
    def batchmatmul(args: list, node):
        assert len(args) == 2
        x = copy.deepcopy(args[0])
        y = copy.deepcopy(args[1])
        x.size = x.size[1:]
        y.size = y.size[1:]
        return InferValue.matmul([x, y], node)

    @staticmethod
    def batchtospacend(args: list, node):
        assert len(args) == 3
        return args[0].value

    @staticmethod
    def spacetobatchnd(args: list, node):
        assert len(args) == 3
        return args[0].value

    @staticmethod
    def biasadd(args: list, node):
        assert len(args) == 2 and len(args[1].size) == 1 and (
                str(args[0].size[-1]) == "?" or str(args[1].size[0]) or args[0].size[-1] == args[1].size[0])
        return Range(left=args[0].value.left + args[1].value.left,
                     right=args[0].value.right + args[1].value.right)

    @staticmethod
    def broadcastargs(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def cast(args: list, node):
        # tf.int64: 9; tf.int32: 3; tf.int16: 5; tf.int8: 6; 
        # tf.uint64: 23; tf.uint32: 22; tf.uint16: 17; tf.uint8: 4; 
        # tf.float64 2; tf.float32: 1; tf.float16: 19; 
        # tf.bool: 10; 
        assert len(args) == 1
        bool_proto = [10]
        int_proto = [9, 3, 5, 6] + [23, 22, 17, 4]
        float_proto = [2, 1, 19]
        attrs = node.attr
        if int(attrs['SrcT'].type) in bool_proto and int(attrs['DstT'].type) in int_proto + float_proto:
            return Range(left=0, right=1)
        elif int(attrs['SrcT'].type) in int_proto + float_proto and int(attrs['DstT'].type) in [10]:
            return Range(left=False, right=True)
        elif int(attrs['SrcT'].type) in int_proto and int(attrs['DstT'].type) in int_proto:
            return args[0].value
        elif int(attrs['SrcT'].type) in float_proto and int(attrs['DstT'].type) in float_proto:
            return args[0].value
        elif int(attrs['SrcT'].type) in int_proto and int(attrs['DstT'].type) in float_proto:
            return args[0].value
        elif int(attrs['SrcT'].type) in float_proto and int(attrs['DstT'].type) in int_proto:
            return InferValue.floor(args, node)
        else:
            raise NotImplementedError("%s -> %s not implemented!" % (attrs['SrcT'].type, attrs['DstT'].type))

    @staticmethod
    def checknumerics(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def cholesky(args: list, node):
        return dumy()

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
            return np.minimum(np.maximum(args[0].value, args[1].value), args[2].value)

    @staticmethod
    def concatv2(args: list, node):
        any_range = False
        for x in args:
            if isinstance(x.value, Range):
                any_range = True
                break

        if not any_range:
            return np.concatenate([x.value for x in args[:-1]], axis=np.int32(args[-1].value))
        else:
            return packtorange(args[:-1], node)

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
        x = identity([args[0]], node)
        y = identity([args[1]], node)
        ends = [x.left * y.left * ind, x.left * y.right * ind,
                x.right * y.left * ind, x.right * y.right * ind]
        return Range(left=min(ends), right=max(ends))

    @staticmethod
    def conv2dbackpropinput(args: list, node):
        return Range(left=-1, right=1)
        return getattr(parse_format_text, "variablev2")(node)

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
    def diag(args: list, node):
        assert len(args) == 1
        tmp = packtorange(args, node)
        return Range(left=min(0, tmp.left), right=max(0, tmp.right))

    @staticmethod
    def dynamicstitch(args: list, node):
        assert len(args) % 2 == 0
        datas = args[len(args) // 2:]
        return packtorange(datas, node)

    @staticmethod
    def enter(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def equal(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def exit(args: list, node):
        return InferValue.identity(args, node)

    @staticmethod
    def expanddims(args: list, node):
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            return np.expand_dims(args[0].value, axis=np.int32(args[1].value))
        else:
            return identity(args, node)

    @staticmethod
    def fifoqueuev2(args: list, node):
        return InferValue.randomshufflequeuev2(args, node)

    @staticmethod
    def fill(args: list, node):
        assert len(args) == 2
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            ret = np.empty(args[0].value)
            ret.fill(args[1].value)
            return ret
        else:
            return identity([args[1]])

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

        x = identity([args[0]], node)
        mean = identity([args[1]], node)
        variance = identity([args[2]], node) + epsilon

        if not is_training:
            offset = identity([args[3]], node)
            scale = identity([args[4]], node)
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
    def gathernd(args: list, node):
        assert len(args) == 2
        return identity(args, node)

    @staticmethod
    def gatherv2(args: list, node):
        assert len(args) == 3
        return identity(args, node)

    @staticmethod
    def greater(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def greaterequal(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def identity(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def isfinite(args: list, node):
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
    def leakyrelu(args: list, node):
        assert len(args) == 1
        alpha = node.attr["alpha"].f

        def leaky_relu(x):
            if x >= 0:
                return x
            else:
                return alpha * x

        if isinstance(args[0].value, Range):
            return Range(left=leaky_relu(args[0].value.left), right=leaky_relu(args[0].value.right))
        else:
            return leaky_relu(args[0].value)

    @staticmethod
    def l2loss(args: list, node):
        assert len(args) == 1
        return InferValue.square(args, node) * 0.5

    @staticmethod
    def less(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def lessequal(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def lgamma(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            ends = [safelgamma(args[0].value.left), safelgamma(args[0].value.right)]
            return Range(left=min(ends), right=max(ends))
        else:
            return safelgamma(args[0].value)

    @staticmethod
    def linspace(args: list, node):
        assert len(args) == 3
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range) or isinstance(args[2].value, Range):
            return packtorange(args[:-1], node)
        else:
            return np.linspace(args[0].value, args[1].value, args[2].value)

    @staticmethod
    def logicaland(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def logicalnot(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

    @staticmethod
    def logicalor(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
        raise NotImplementedError

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
        assert len(args) == 2
        try:
            len(args[0].size) == len(args[1].size)
        except:
            return dumy()
        assert len(args[0].size) == len(args[1].size)
        for i in range(len(args[0].size) - 2):
            assert str(args[0].size[i]) == "?" or str(args[1].size[i]) == "?" or args[0].size[i] == args[1].size[i]
        ind = real_size(args[0].size[-1], args[1].size[-2])
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            return np.matmul(args[0].value, args[1].value)
        else:
            x = identity([args[0]], node)
            y = identity([args[1]], node)
            ends = [x.left * y.left * ind, x.left * y.right * ind, x.right * y.left * ind, x.right * y.right * ind]
            return Range(left=min(ends), right=max(ends))

    @staticmethod
    def matrixdiag(args: list, node):
        assert len(args) == 1
        tmp = packtorange(args, node)
        return Range(left=min(0, tmp.left), right=max(0, tmp.right))

    @staticmethod
    def matrixbandpart(args: list, node):
        assert len(args) == 3
        tmp = packtorange(args[:1], node)
        return Range(left=min(tmp.left, 0), right=max(tmp.right, 0))

    @staticmethod
    def matrixdiagpart(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def max(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def maxpool(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def maximum(args: list, node):
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
    def mean(args: list, node):
        assert len(args) == 2
        return identity(args, node)

    @staticmethod
    def merge(args: list, node):
        tmp = packtorange(args, node)
        max_index = int(node.attr["N"].i)
        return_index = Range(left=0, right=max_index - 1)
        if isinstance(tmp, tuple):
            raise AssertionError
        else:
            return [tmp, return_index]

    @staticmethod
    def min(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def minimum(args: list, node):
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
    def mul(args: list, node):
        assert len(args) == 2
        if args[0].value is None or args[1].value is None:
            return None
        if isinstance(args[1].value, Range) or isinstance(args[0].value, Range):
            x = identity([args[0]], node)
            y = identity([args[1]], node)
            ends = [x.left * y.left, x.left * y.right, x.right * y.left, x.right * y.right]
            return Range(left=min(ends), right=max(ends))
        else:
            return args[0].value * args[1].value

    def multinomial(args: list, node):
        assert len(args) == 2
        return Range(left=0, right=1)

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
            return Range(left=0, right=length_unknown)

    @staticmethod
    def notequal(args: list, node):
        if not turn_on_bool:
            return Range(left=False, right=True)
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
        any_range = False
        for x in args:
            if isinstance(x.value, Range):
                any_range = True
                break

        if not any_range:
            return np.stack([x.value for x in args], axis=int(node.attr["axis"].i))
        else:
            return packtorange(args, node)

    @staticmethod
    def pad(args: list, node):
        return identity(args, node)

    @staticmethod
    def paddingfifoqueuev2(args: list, node):
        return InferValue.randomshufflequeuev2(args, node)

    @staticmethod
    def parsesingleexample(args: list, node):
        assert len(args) == 3
        return [Range(left=0, right=length_unknown) for _ in range(20)]

    @staticmethod
    def placeholder(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, node.op.lower())(node)

    @staticmethod
    def placeholderwithdefault(args: list, node):
        assert len(args) == 1
        tmp = getattr(parse_format_text, 'placeholder')(node)
        if isinstance(args[0].value, Range):
            return Range(left=min(args[0].value.left, tmp.left), right=max(args[0].value.right, tmp.right))
        else:
            return Range(left=min(args[0].value, tmp.left), right=max(args[0].value, tmp.right))

    @staticmethod
    def pow(args: list, node):
        assert len(args) == 2
        if isinstance(args[0].value, Range) and isinstance(args[1].value, Range):
            return Range(left=safepow(args[0].value.left, args[1].value.left),
                         right=safepow(args[0].value.right, args[1].value.right))
        elif isinstance(args[0].value, Range):
            return Range(left=safepow(args[0].value.left, args[1].value),
                         right=safepow(args[0].value.right, args[1].value))
        elif isinstance(args[1].value, Range):
            return Range(left=safepow(args[0].value, args[1].value.left),
                         right=safepow(args[0].value, args[1].value.right))
        else:
            return safepow(args[0].value, args[1].value)

    @staticmethod
    def prod(args: list, node):
        assert len(args) == 2
        if args[0].value is None:
            return None
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            try:
                ind = int(args[0].size[int(args[1].value)])
                return Range(left=safepow(args[0].value.left, ind), right=safepow(args[0].value.right, ind))
            except:
                ind = Range(left=0, right=length_unknown)
                t = InferValue.pow([args[0], AbstractInterpretation(value=ind, dtype=3, size=[])], node)
                if isinstance(t, tuple):
                    raise AssertionError
                else:
                    return t
        else:
            axises = np.int32(args[1].value)
            return np.prod(args[0].value, axis=tuple(axises) if len(axises.shape) > 0 else axises)

    @staticmethod
    def queuedequeuemanyv2(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def randomshuffle(args: list, node):
        assert len(args) == 1
        return identity(args, node)

    @staticmethod
    def randomshufflequeuev2(args: list, node):
        assert len(args) == 0
        return getattr(parse_format_text, "oneshotiterator")(node)

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
        all_single_np = True
        for arg in args:
            if isinstance(arg.value, Range) or len(np.array(arg.value).shape) > 0:
                all_single_np = False
                break

        if not all_single_np:
            left = args[0].value.left if isinstance(args[0].value, Range) else np.min(args[0].value)
            right = args[1].value.right if isinstance(args[1].value, Range) else np.max(args[1].value)
            return Range(left=left, right=right)
        else:
            return np.arange(args[0].value, args[1].value, args[2].value)

    @staticmethod
    def rank(args: list, node):
        assert len(args) == 1
        try:
            return int(args[0].size)
        except:
            return Range(left=1, right=length_unknown)

    @staticmethod
    def readvariableop(args: list, node):
        assert len(args) == 1
        return args[0].value

    @staticmethod
    def realdiv(args: list, node):
        assert len(args) == 2
        x = args[0].value
        y = args[1].value
        if not isinstance(x, Range):
            x = np.reshape(x, -1)
        if not isinstance(y, Range):
            y = np.reshape(y, -1)
        if isinstance(x, Range) and isinstance(y, Range):
            if y.left > 0 or y.right < 0:
                ends = [x.left / y.left, x.left / y.right, x.right / y.left, x.right / y.right]
                return Range(left=np.min(ends), right=np.max(ends))
            else:
                return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)
        elif not isinstance(y, Range):  # x can be a Range or a np.array
            if isinstance(x, Range):
                ends = [x.left / yy for yy in y] + [x.right / yy for yy in y]
                return Range(left=np.min(ends), right=np.max(ends))
            else:
                return x * (1 / y)
        else:  # if y is a Range, whatever x is, we have to end up with a Range, but we can do it precisely when x is a float
            if y.left > 0 or y.right < 0:
                ends = [xx / y.left for xx in x] + [xx / y.right for xx in x]
                return Range(left=np.min(ends), right=np.max(ends))
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
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            return np.reshape(args[0].value, np.int32(args[1].value))
        else:
            return identity(args, node)

    @staticmethod
    def resizearea(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def resizebilinear(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def resizenearestneighbor(args: list, node):
        assert len(args) == 2
        return args[0].value

    @staticmethod
    def resourcegather(args: list, node):
        assert len(args) == 2
        return identity(args, node)

    @staticmethod
    def reversev2(args: list, node):
        assert len(args) == 2
        return identity(args, node)

    @staticmethod
    def round(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=np.round(args[0].value.left), right=np.round(args[0].value.right))
        return np.round(args[0].value)

    @staticmethod
    def rsqrt(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left = safesqrt(args[0].value.left)
            right = safesqrt(args[0].value.right)
            if left == 0 or right == 0:
                return dumy()
            else:
                return Range(left=1 / right, right=1 / left)
        else:
            return 1 / safesqrt(args[0].value)

    @staticmethod
    def select(args: list, node):
        assert len(args) == 3
        if not isinstance(args[0].value, Range):
            raise NotImplementedError("not implemented when the condition is known")

        x = identity([args[1]], node)
        y = identity([args[2]], node)
        if not turn_on_bool:
            return Range(left=min(x.left, y.left), right=max(x.right, y.right))
        raise NotImplementedError

    @staticmethod
    def shape(args: list, node):
        assert len(args) == 1
        try:
            return [int(x) for x in args[0].size]
        except:
            return Range(left=1, right=length_unknown)

    @staticmethod
    def sign(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=np.sign(args[0].value.left), right=np.sign(args[0].value.right))
        else:
            return np.sign(args[0].value)

    @staticmethod
    def size(args: list, node):
        assert len(args) == 1
        try:
            ele = 1
            for x in args[0].size:
                ele *= int(x)
            if ele < 0:
                return Range(left=0, right=length_unknown)
            else:
                return ele
        except:
            return Range(left=0, right=length_unknown)

    @staticmethod
    def slice(args: list, node):
        assert len(args) == 3
        try:
            return args[0].value[
                tuple(slice(a, a + b) if b >= 0 else slice(a, None) for a, b in zip(args[1].value, args[2].value))]
        except:
            return identity(args, node)

    @staticmethod
    def sparsetodense(args: list, node):
        assert len(args) == 4
        return Range(left=0, right=1)

    @staticmethod
    def split(args: list, node):
        assert len(args) == 2
        nums = int(node.attr["num_split"].i)
        if nums == 1:
            return identity(args[1:], node)
        else:
            return [identity(args[1:], node) for _ in range(nums)]

    @staticmethod
    def sqrt(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            left = safesqrt(args[0].value.left)
            right = safesqrt(args[0].value.right)

            return Range(left=left, right=right)
        else:
            return safesqrt(args[0].value)

    @staticmethod
    def square(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            abs_value = InferValue.abs(args, node)
            return Range(left=abs_value.left * abs_value.left, right=abs_value.right * abs_value.right)
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
        return identity(args, node)

    @staticmethod
    def stopgradient(args: list, node):
        return InferValue.identity(args, node)

    @staticmethod
    def stridedslice(args: list, node):
        return identity(args, node)

    @staticmethod
    def sub(args: list, node):
        assert len(args) == 2
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            x = identity([args[0]], node)
            y = identity([args[1]], node)
            return Range(left=x.left - y.right, right=x.right - y.left)
        else:
            return args[0].value - args[1].value

    @staticmethod
    def sum(args: list, node):
        assert len(args) == 2
        if args[0].value is None:
            return None
        if isinstance(args[0].value, Range) or isinstance(args[1].value, Range):
            try:
                ind = int(args[0].size[int(args[1].value)])
                return Range(left=args[0].value.left * ind, right=args[0].value.right * ind)
            except:
                ind = Range(left=1, right=1e6)
                t = InferValue.mul([args[0], AbstractInterpretation(value=ind, dtype=3, size=[])], node)
                if isinstance(t, tuple):
                    raise AssertionError
                else:
                    return t
        else:
            axises = np.int32(args[1].value)
            return np.sum(args[0].value, axis=tuple(axises) if len(axises.shape) > 0 else axises)

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
        if not isinstance(args[0].value, Range) and not isinstance(args[1].value, Range):
            return np.tile(args[0].value, np.int32(args[1].value))
        else:
            return identity(args, node)

    @staticmethod
    def topkv2(args: list, node):
        assert len(args) == 2
        try:
            ind = int(args[0].size[-1])
            value = Range(left=0, right=ind - 1)
        except:
            value = Range(left=0, right=length_unknown)
        return [identity(args, node), value]

    @staticmethod
    def transpose(args: list, node):
        assert len(args) == 2
        try:
            return np.transpose(args[0].value, np.int32(args[1].value))
        except:
            return identity(args, node)

    @staticmethod
    def unpack(args: list, node):
        assert len(args) == 1
        nums = int(node.attr["num"].i)
        axis = int(node.attr["axis"].i)
        if not isinstance(args[0].value, Range):
            assert args[0].value.shape[axis] == nums
            if nums == 1:
                index = [slice(None) for _ in range(len(args[0].value.shape))]
                index[axis] = 0
                return args[0].value[index]
            else:
                ret = []
                for i in range(nums):
                    index = [slice(None) for _ in range(len(args[0].value.shape))]
                    index[axis] = i
                    ret.append(args[0].value[index])

                return ret
        else:
            if nums == 1:
                return identity(args, node)
            else:
                return [identity(args, node) for _ in range(nums)]

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
            return Range(left=0, right=length_unknown - 1)

    @staticmethod
    def zeroslike(args: list, node):
        assert len(args) == 1
        try:
            if len(args[0].size) == 0:
                return 0
        except:
            pass

        return Range(left=0, right=0)

    @staticmethod
    def floormod(args: list, node):
        def mod(x, y):
            return x - math.floor(x / y) * y

        assert len(args) == 2
        try:
            x = float(args[0].value)
        except:
            x = identity([args[0]], node)
        try:
            y = float(args[1].value)
        except:
            y = identity([args[1]], node)

        if isinstance(x, Range) and isinstance(y, Range):
            if y.left > 0 or y.right < 0:
                ends = [mod(x.left, y.left), mod(x.left, y.right), mod(x.right, y.left), mod(x.right, y.right)]
                return Range(left=min(ends), right=max(ends))
            else:
                return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)
        elif not isinstance(y, Range):
            return x * (1 / y)
        else:
            if y.left > 0 or y.right < 0:
                ends = [mod(x, y.left), mod(x, y.right)]
                return Range(left=min(ends), right=max(ends))
            else:
                return Range(left=-OVERFLOW_LIMIT, right=OVERFLOW_LIMIT)

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
    def sin(args: list, node):
        assert len(args) == 1
        return Range(left=-1, right=1)

    def cos(args: list, node):
        assert len(args) == 1
        return Range(left=-1, right=1)

    @staticmethod
    def log(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            if args[0].value.left <= 0:
                return Range(left=-OVERFLOW_LIMIT, right=math.log(args[0].value.right))
            else:
                return Range(left=math.log(args[0].value.left), right=math.log(args[0].value.right))
        else:
            return np.log(args[0].value)

    @staticmethod
    def log1p(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            if args[0].value.left <= -1:
                return Range(left=-OVERFLOW_LIMIT, right=np.log1p(args[0].value.right))
            else:
                return Range(left=np.log1p(args[0].value.left), right=np.log1p(args[0].value.right))
        else:
            return np.log1p(args[0].value)

    @staticmethod
    def softplus(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=safesoftplus(args[0].value.left), right=safesoftplus(args[0].value.right))
        else:
            return safesoftplus(args[0].value)

    @staticmethod
    def exp(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=safeexp(args[0].value.left), right=safeexp(args[0].value.right))
        else:
            return safeexp(args[0].value)

    @staticmethod
    def softmax(args: list, node):
        assert len(args) == 1
        try:
            ind = int(args[0].size[-1])
        except:
            ind = None

        if isinstance(args[0].value, Range):
            min_ele = safeexp(args[0].value.left)
            max_ele = safeexp(args[0].value.right)
            if max_ele >= OVERFLOW_LIMIT or min_ele == 0:
                left = 0
            elif ind is not None:
                left = min_ele / ((ind - 1) * max_ele + min_ele)
            else:
                left = min_ele / ((length_unknown - 1) * max_ele + min_ele)
            if max_ele >= OVERFLOW_LIMIT or min_ele == 0:
                right = 1
            elif ind is not None:
                right = max_ele / ((ind - 1) * min_ele + max_ele)
            else:
                right = max_ele / (min_ele + max_ele)
            return Range(left=left, right=right)
        else:
            tmp_exp = np.exp(args[0].value)
            return tmp_exp / np.sum(tmp_exp)

    @staticmethod
    def sigmoid(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=1 / (1 + safeexp(-args[0].value.left)), right=1 / (1 + safeexp(-args[0].value.right)))
        else:
            return 1 / (1 + safeexp(-args[0].value))

    @staticmethod
    def tanh(args: list, node):
        assert len(args) == 1
        if isinstance(args[0].value, Range):
            return Range(left=np.tanh(args[0].value.left), right=np.tanh(args[0].value.right))
        else:
            return np.tanh(args[0].value)


# contains the abstract interpretations of TensorFlow APIs used in the tensor partition and the linear affine relation.
class InferArray:
    @staticmethod
    def add(args: list, node):
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
        try:
            len(args[0].size) == len(args[1].size)
        except:
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
        if len(ret.block_to_symbol.keys()) == 0:
            return None
        x = list(ret.block_to_symbol.keys())[0]
        ret.block_to_symbol[x].value = {}
        ret.block_to_symbol[x].map_to_index = {}

        return ret

    @staticmethod
    def relu(args: list, node):
        # right now it will abort when it encounters relu(z=x-y). 
        # A better approach is to set it to relu(z) instead of aborting.
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

    @staticmethod
    def pack(args: list, node):
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
