class AbstractInterpretation:
    def __init__(self, size=None, value=None, dtype=None, constraints=None, array=None):
        # the shape of the tensor extracted from the protocal buffer file.
        self.size = size
        # the interval abstraction stored in a Range object or a numpy concrete value.
        self.value = value
        self.constraints = constraints
        # the data type of the tensor extracted from the protocal buffer file.
        self.dtype = dtype
        # the tensor partition stored in a Array object.
        self.array = array

    # check whether some of the fields are None, which indicates that dataflow analysis cannot infer this abstracted
    # value due to unimplemented TensorFlow APIs.
    def has_none(self):
        return self.size is None or self.value is None or self.dtype is None

    # gets the i-th index of all the fields and returns a new AbstractInterpretation object.
    # returns self if i is None.
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
