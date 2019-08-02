import z3
from utils import resolve_type
import math
import numpy as np
from itertools import product


class Solver:
    solver = z3.Solver()
    index = {}
    variable_by_name = {}

    @staticmethod
    def add_variable(name, dtype):
        if name not in Solver.index:
            Solver.index[name] = 0
        variable_name = name + "_" + str(Solver.index[name])
        Solver.index[name] += 1
        if dtype in [3]:
            real_name = variable_name + "_Int"
            Solver.variable_by_name[real_name] = z3.Int(real_name)
        elif dtype in [1]:
            real_name = variable_name + "_Real"
            Solver.variable_by_name[real_name] = z3.Real(real_name)
        elif dtype in [10]:
            real_name = variable_name + "_Bool"
            Solver.variable_by_name[real_name] = z3.Bool(real_name)
        else:
            raise NotImplementedError("Cannot Recognize: ", dtype)
        return Solver.variable_by_name[real_name]

    @staticmethod
    def max(x, ys_):
        ys1 = [y for y in list(map(resolve_type, ys_)) if str(y) != 'inf']
        ys = [y for y in ys1 if str(y) != '-inf']
        if len(ys1) != len(ys_):
            z3.And([x >= y for y in ys])

        try:
            return x == max(ys)
        except:
            pass
        if len(ys) == 1:
            return x == ys[0]
        if len(ys) == 2:
            return x == z3.If(ys[0] > ys[1], ys[0], ys[1])
        return z3.And(z3.Or([x == y for y in ys]), z3.And([x >= y for y in ys]))

    @staticmethod
    def min(x, ys_):
        ys1 = [y for y in list(map(resolve_type, ys_)) if str(y) != '-inf']
        ys = [y for y in ys1 if str(y) != 'inf']
        if len(ys1) != len(ys_):
            z3.And([x <= y for y in ys])

        try:
            return x == min(ys)
        except:
            pass
        if len(ys) == 1:
            return x == ys[0]
        if len(ys) == 2:
            return x == z3.If(ys[0] < ys[1], ys[0], ys[1])
        return z3.And(z3.Or([x == y for y in ys]), z3.And([x <= y for y in ys]))

    @staticmethod
    def in_interval(x, interval):
        if isinstance(interval, tuple):
            if interval[0] > 0 or interval[1] > 0:
                # (a, b]
                if math.isinf(interval[1]):
                    return z3.And(interval[0] < x)
                else:
                    return z3.And(interval[0] < x, x <= interval[1])
            else:
                # [a, b)
                if math.isinf(interval[0]):
                    return z3.And(x < interval[1])
                else:
                    return z3.And(interval[0] <= x, x < interval[1])
        else:
            return x == interval


class Range:
    def __init__(self, *args, **kwargs):
        """Two ways of construction:
                left, right
                name, dtype
            One optional parameter for range_const

            The int and float tensor representation --> interval
            The bool tensor representation -->  [True, False] for all False,
                                                [False, True] for all True,
                                                [True, True] for both True and False
        """
        if "const_type" in kwargs:
            self.const_type = kwargs["const_type"]
        else:
            self.const_type = None
        if "name" in kwargs and "dtype" in kwargs:
            name = kwargs["name"]
            dtype = kwargs["dtype"]
            self.left = Solver.add_variable(name + "L", dtype)
            self.right = Solver.add_variable(name + "R", dtype)
        elif "left" in kwargs and "right" in kwargs:
            self.left = resolve_type(kwargs["left"])
            self.right = resolve_type(kwargs["right"])
        else:
            raise NotImplementedError(args, kwargs, " setting not implemented")

    def __str__(self):
        return "[%s, %s]\n[%s, %s]" % (self.left, self.right, str(type(self.left)), str(type(self.right)))

    def __repr__(self):
        return "[%s, %s]\n[%s, %s]" % (self.left, self.right, str(type(self.left)), str(type(self.right)))

    def __mul__(self, other):
        return Range(left=None if self.left is None else self.left * other,
                     right=None if self.right is None else self.right * other,
                     const_type=self.const_type)

    def __add__(self, other):
        return Range(left=None if self.left is None else self.left + other,
                     right=None if self.right is None else self.right + other,
                     const_type=self.const_type)


class Array:
    def __init__(self, symbol, size):
        self.index_slices = []
        for i in range(len(size)):
            try:
                self.index_slices.append([int(size[i])])
            except:
                self.index_slices.append(None)
        self.block_to_symbol = {tuple([None if x is None else x[0] for x in self.index_slices]): symbol}

    @staticmethod
    def join_index_slices(a, b):
        ret = []
        for i in range(len(a)):
            if a[i] is None and b[i] is None:
                ret.append(None)
            elif a[i] is None:
                ret.append(b[i])
            elif b[i] is None:
                ret.append(a[i])
            else:
                c = np.unique(a + b)
                ret.append(list(c))

        return ret

    def get_corresponding_keys(self, index_slices):
        iargs = [None if x is None else 0 for x in self.index_slices]
        ret = []
        for indexes in product(*index_slices):
            key = ()
            for i in range(len(indexes)):
                if indexes[i] is not None:
                    while indexes[i] > self.index_slices[i][iargs[i]]:
                        iargs[i] += 1

                    key += (self.index_slices[i][iargs[i]],)
                else:
                    key += (None,)

            ret.append(self.block_to_symbol[key])

        return ret

    def get_possible_values(self):
        ret = []
        for ix in self.block_to_symbol:
            flag = True
            x = self.block_to_symbol[ix]
            for y in ret:
                if y.equals(x):
                    flag = False
                    break

            if flag:
                ret.append(x)

        return [str_2_linear_expression(str(x)) for x in ret]

    def __str__(self):
        ret_str = ""
        for x in self.block_to_symbol:
            ret_str += x + "\t" + str(self.block_to_symbol[x]) + "\n"
        return ret_str

    def __repr__(self):
        ret_str = ""
        for x in self.block_to_symbol:
            ret_str += x + "\t" + str(self.block_to_symbol[x]) + "\n"
        return ret_str


def check_range_const(range_const: Range):
    if z3.is_arith(range_const.left) or z3.is_arith(range_const.right):
        return True
    return not (range_const.left is not None and range_const.right is not None and range_const.left > range_const.right)


def meet(range, range_const: Range):
    if not check_range_const(range_const):
        return False
    assert range_const.const_type is not None

    if range_const.const_type == 0:
        if isinstance(range, Range):
            if range_const.left is not None and range_const.right is not None:
                return z3.Not(z3.Or(range_const.right < range.left, range.right < range_const.left))
            if range_const.right is not None:
                return z3.Or(range.left <= range_const.right, range.right <= range_const.right)
            if range_const.left is not None:
                return z3.Or(range_const.left <= range.left, range_const.left <= range.right)
            else:
                return True
        else:
            if range_const.left is not None and range_const.right is not None:
                return bool(np.all(range_const.left <= range) and np.all(range <= range_const.right))
            if range_const.right is not None:
                return bool(np.all(range <= range_const.right))
            if range_const.left is not None:
                return bool(np.all(range_const.left <= range))
            else:
                return True
    else:
        if isinstance(range, Range):
            if range_const.left is not None and range_const.right is not None:
                return z3.And(range_const.left >= range.left, range.right >= range_const.right)
            else:
                return False
        else:
            return range_const.left == range and range == range_const.right


def str_2_linear_expression(x: str):
    ret = []
    x += " "
    if x[0] != '-' and x[0] != '+':
        x = "+" + x
    sign = 1
    factor = 0
    symbol = ""
    has_sign = False
    has_factor = False
    i = 0
    while i < len(x):
        if x[i] in ['-', '+']:
            if x[i] == '-':
                sign = -1
            else:
                sign = 1
            has_sign = True
        elif has_factor:
            if x[i] == ' ':
                ret.append((sign * factor, symbol))
                sign = 1
                factor = 0
                symbol = ""
                has_sign = False
                has_factor = False
            else:
                symbol += x[i]
        elif has_sign:
            if '0' <= x[i] <= '9':
                factor = factor * 10 + ord(x[i]) - 48
            elif x[i] == '*':
                has_factor = True
            elif x[i] != ' ':
                factor = 1
                has_factor = True
                symbol += x[i]

        i += 1

    return ret
