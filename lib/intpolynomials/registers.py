from copy import copy

import numpy as np
from intpolynomials import Int_Polynomial_Array

def is_int(num):
    return isinstance(num, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64))

try:
    from cornifer import Numpy_Register, Register

except ModuleNotFoundError:
    Int_Polynomial_Register = None

else:

    class Int_Polynomial_Register(Numpy_Register):

        @classmethod
        def dump_disk_data(cls, data, filename, **kwargs):
            return cls.__bases__[0].dump_disk_data(data.get_ndarray(), filename, **kwargs)

        @classmethod
        def load_disk_data(cls, filename, **kwargs):

            data, filename = cls.__bases__[0].load_disk_data(filename, **kwargs)
            array = Int_Polynomial_Array(data.shape[1] - 1)
            array.set(data)
            return array, filename


    Register.add_subclass(Int_Polynomial_Register)


