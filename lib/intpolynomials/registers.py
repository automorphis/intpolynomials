from copy import copy

import numpy as np
from intpolynomials import IntPolynomialArray

def is_int(num):
    return isinstance(num, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64))

try:
    from cornifer import NumpyRegister, Register

except ModuleNotFoundError:
    IntPolynomialRegister = None

else:

    class IntPolynomialRegister(NumpyRegister):

        @classmethod
        def dump_disk_data(cls, data, filename, **kwargs):
            cls.__bases__[0].dump_disk_data(data.get_ndarray(), filename, **kwargs)

        @classmethod
        def load_disk_data(cls, filename, **kwargs):

            data = cls.__bases__[0].load_disk_data(filename, **kwargs)
            array = IntPolynomialArray(data.shape[1] - 1)
            array.set(data)
            return array


