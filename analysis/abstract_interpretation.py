import parse.parse_format_text as parse_format_text
import warnings


class AbstractInterpretation:
    def __init__(self, size=None, value=None, dtype=None, constraints=None, node=None):
        self.size = size
        self.value = value
        self.constraints = constraints
        self.dtype = dtype

    def has_none(self):
        return self.size is None or self.value is None or self.dtype is None

    def index_of(self, i):
        if i is None:
            return self
        if self.has_none():
            return AbstractInterpretation()

        return AbstractInterpretation(size=self.size[i], value=self.value[i], dtype=self.dtype[i],
                                      constraints=self.constraints)


    def __str__(self):
        return "size: %s\nvalue: %s\ndtype: %s\nconstraints: %s\n" % (
            str(self.size), str(self.value), str(self.dtype), str(self.constraints))

    # def __iand__(self, other):
    #     self.size = other.size
    #     self.value = other.value
    #     self.dtype = other.dtype
    #     self.constraints = other.constraints
