import numpy as np

OVERFLOW_LIMIT = 1e38
UNDERFLOW_LIMIT = 1e-37
OVERFLOW_D = 38
UNDERFLOW_D = -37


def resolve_type(y):
    if isinstance(y, np.int32) or isinstance(y, np.int64):
        return int(y)
    elif isinstance(y, np.float32) or isinstance(y, np.float64):
        return float(y)
    elif isinstance(y, np.bool):
        return bool(y)
    else:
        return y

def shape_from_proto(shape):
    s = str(shape)
    x = 0
    u = []
    for i in range(len(s)):
        if s[i] >= '0' and s[i] <= '9':
            x = x * 10 + ord(s[i]) - 48
        elif x != 0:
            u.append(x)
            x = 0
    
    return u
