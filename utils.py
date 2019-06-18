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
