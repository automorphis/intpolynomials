from copy import copy

import numpy as np
from intpolynomials import Int_Polynomial_Array

def is_int(num):
    return isinstance(num, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64))

try:
    from cornifer import NumpyRegister, Register

except ModuleNotFoundError:
    Int_Polynomial_Register = None

else:

    class Int_Polynomial_Register(NumpyRegister):

        @classmethod
        def dumpDiskData(cls, data, filename, **kwargs):
            cls.__bases__[0].dumpDiskData(data.get_ndarray(), filename, **kwargs)

        @classmethod
        def loadDiskData(cls, filename, **kwargs):

            data = cls.__bases__[0].loadDiskData(filename, **kwargs)
            array = Int_Polynomial_Array(data.shape[1] - 1)
            array.set(data)
            return array

    Register.addSubclass(Int_Polynomial_Register)


