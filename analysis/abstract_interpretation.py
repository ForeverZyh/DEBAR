import parse.parse_format_text as parse_format_text
import warnings


class AbstractInterpretation:
    def __init__(self, size=None, value=None, dtype=None, constraints=None, array=None):
        self.size = size
        self.value = value
        self.constraints = constraints
        self.dtype = dtype
        self.array = array

    def has_none(self):
        return self.size is None or self.value is None or self.dtype is None

    def index_of(self, i):
        if i is None:
            return self
        if self.has_none():
            return AbstractInterpretation()

        return AbstractInterpretation(size=self.size[i], value=self.value[i], dtype=self.dtype[i],
                                      constraints=self.constraints, array=self.array[i])

    def __str__(self):
        return "size: %s\nvalue: %s\ndtype: %s\nconstraints: %s\n array blocks: %s\n" % (
            str(self.size), str(self.value), str(self.dtype), str(self.constraints), str(self.array))
