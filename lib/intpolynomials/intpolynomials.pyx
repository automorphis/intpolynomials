from intpolynomials cimport *

import warnings

import numpy as np
import cython
from mpmath import workdps, mpf

COEF_DTYPE = np.int64

cpdef instantiate_int_poly(DEG_t deg, DPS_t dps, is_natural = True):
    return Int_Polynomial(
        np.zeros(deg + 1, dtype=COEF_DTYPE),
        dps,
        is_natural
    )

cdef inline object cpb(BOOL_TYPE b):
    return True if b == TRUE else False

cdef inline BOOL_TYPE pcb(object b):
    return TRUE if b else FALSE

cdef class Int_Polynomial:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _init(self, coefs, DPS_t dps, BOOL_TYPE is_natural) except *:
        cdef cnp.ndarray[COEF_t, ndim=1] coefs_array
        cdef DEG_t i

        if isinstance(coefs, list) or isinstance(coefs, tuple):
            coefs_array = np.array(coefs, dtype=COEF_DTYPE)

        elif not isinstance(coefs, np.ndarray):
            raise TypeError("passed coefs must be either list, tuple, or np.ndarray. passed type: %s" % type(coefs))

        elif coefs.dtype != COEF_DTYPE:
            warnings.warn("Int_Polynomial constructor: Automatically casting to np.longlong is dangerous. Please cast to " +
                            "np.longlong prior to calling this constructor.")
            coefs_array = coefs.astype(COEF_DTYPE)

        else:
            coefs_array = coefs

        if len(coefs_array) == 0:
            raise ValueError("passed coefficient array must be non-empty; pass [0] for the zero polynomial")
        else:
            self._coefs = coefs_array

        self._dps = dps
        self._deg = <DEG_t> len(self._coefs) - 1
        self._max_deg = self._deg
        self._is_natural = is_natural
        with workdps(self._dps):
            self.last_eval = mpf(0)
            self.last_eval_deriv = mpf(0)
        self._start_index = 0

        self.trim()

    def __init__(self, coefs, dps, is_natural = True):
        self._init(coefs, dps, pcb(is_natural))

    def __copy__(self):
        return Int_Polynomial(self.ndarray_coefs(cpb(self._is_natural),True), self.get_dps(), cpb(self._is_natural))

    def __deepcopy__(self,memo):
        return self.__copy__()

    cpdef Int_Polynomial trim(self):

        cdef DEG_t less = 0
        cdef DEG_t deg = self.get_deg()
        cdef DEG_t i

        if deg < 0:
            return self

        for i in range(deg + 1):
            if self.get_coef(deg - i) == 0:
                less += 1
            else:
                break

        self._deg -= less
        if self._is_natural == FALSE:
            self._start_index += less

        return self

    cpdef DPS_t get_dps(self):
        return self._dps

    def set_dps(self, dps):
        self._dps = dps

    cpdef DEG_t get_max_deg(self):
        return self._max_deg

    cpdef DEG_t get_deg(self):
        return self._deg

    cdef COEF_t[:] get_coefs_mv(self):
        return self._coefs

    cpdef cnp.ndarray[COEF_t, ndim=1] ndarray_coefs(self, natural_order = True, include_hidden_coefs = False):

        cdef DEG_t i
        cdef DEG_t deg = self.get_deg()
        cdef cnp.ndarray[COEF_t, ndim=1] ret

        if include_hidden_coefs:
            ret = np.empty(max(1, self._max_deg + 1), dtype=COEF_DTYPE)

        else:
            ret = np.empty(max(1, deg + 1), dtype=COEF_DTYPE)

        if deg < 0:
            ret[0] = 0

        elif include_hidden_coefs:

            for i in range(self._max_deg + 1):

                if natural_order:
                    ret[i] = self.get_coef(i)

                else:
                    ret[i] = self.get_coef(self._max_deg - i)

        else:

            for i in range(deg + 1):

                if natural_order:
                    ret[i] = self.get_coef(i)

                else:
                    ret[i] = self.get_coef(deg - i)

        return ret

    cpdef COEF_t max_abs_coef(self):
        cdef INDEX_t i
        cdef DEG_t deg = self.get_deg()
        cdef COEF_t curr_max = -1
        cdef COEF_t c
        for i in range(deg + 1):
            c = self.get_coef(i)
            if c < 0 and -c > curr_max:
                curr_max = -c
            elif c >= 0 and c > curr_max:
                curr_max = c
        return curr_max

    def eval(self, x, calc_deriv = False):
        self._c_eval_both(mpf(x), TRUE if calc_deriv else FALSE)
        if calc_deriv:
            return self.last_eval, self.last_eval_deriv
        else:
            return self.last_eval

    cdef void _c_eval_both(self, MPF_t x, BOOL_TYPE calc_deriv):
        cdef:
            MPF_t p
            MPF_t q
            MPF_t coef
            DEG_t i
            DEG_t deg = self.get_deg()

        if deg < 0:
            p = mpf(0.0)
            q = mpf(0.0)
        else:
            with workdps(self.get_dps()):
                p = mpf(self.get_coef(deg))
                q = mpf(0)
                for i in range(1, deg + 1):
                    if calc_deriv == TRUE:
                        q = p + x*q
                    p = x*p + self.get_coef(deg - i)

        self.last_eval = p
        self.last_eval_deriv = q

    cdef void c_eval_both(self, MPF_t x):
        self._c_eval_both(x, TRUE)

    cdef void c_eval_only(self, MPF_t x):
        self._c_eval_both(x, FALSE)

    def __str__(self):
        return str(list(self.ndarray_coefs()))

    def __repr__(self):
        return (
            "Int_Polynomial(" +
            str(list(self.ndarray_coefs(True, True))) +
            (", %d)" % self.get_dps())
        )

    def __hash__(self):
        cdef int ret = 0
        cdef DEG_t i
        for i in range(self.get_deg() + 1):
            ret += <int> hash(str(self.get_coef(i)))
        return ret + <int> hash(str(self.get_dps()))

    cdef BOOL_TYPE eq(self, Int_Polynomial other):
        cdef DEG_t i
        cdef DEG_t deg = self.get_deg()
        if deg != other.get_deg():
            return FALSE
        for i in range(deg + 1):
            if self.get_coef(i) != other.get_coef(i):
                return FALSE
        return TRUE

    def __getstate__(self):
        return {
            "_coefs": self.ndarray_coefs(cpb(self._is_natural), True),
            "_dps": self.get_dps(),
            "_is_natural": cpb(self._is_natural)
        }

    def __setstate__(self, state):
        self._coefs = state["_coefs"]
        self._deg = len(state["_coefs"]) - 1
        self._max_deg = self._deg
        self._dps = state["_dps"]
        self._is_natural = pcb(state["_is_natural"])
        self._start_index = 0
        self.trim()

    def __ne__(self, other):
        return not(self == other)

    def __eq__(self, other):
        return cpb(self.eq(other))

    def __setitem__(self, i, coef):
        self.set_coef(i, coef)

    def __getitem__(self, i):
        return self.get_coef(i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_coef(self, DEG_t i, COEF_t coef) except *:
        cdef DEG_t deg = self.get_deg()

        if not(0 <= i <= self._max_deg):
            raise IndexError("index must be between 0 and %d. reinitialize array if you want to increase the maximum degree. passed index: %d" % (self._max_deg, i))
        if self._is_natural == TRUE:
            self._coefs[i] = coef
        else:
            self._coefs[deg - i + self._start_index] = coef
        if coef != 0 and i > self._deg:
            self._deg = i
            if self._is_natural == FALSE:
                self._start_index = self._max_deg - i

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COEF_t get_coef(self, DEG_t i) except? -1:

        cdef DEG_t deg = self.get_deg()

        if i < 0:
            raise IndexError("index must be positive or zero.")

        if deg < 0:
            return 0

        if self._is_natural == TRUE:

            if i <= deg:
                return self._coefs[i]

            else:
                return 0

        else:

            if i <= deg:
                return self._coefs[deg - i + self._start_index]

            else:
                return 0

cdef class Int_Polynomial_Array:

    def __init__(self, max_deg, dps):
        self._max_size = 0
        self._curr_index = 0
        self._max_deg = max_deg
        self._dps = dps
        self._array = np.empty((0,0), dtype = COEF_DTYPE)

    cpdef void init_empty(self, INDEX_t init_size) except *:
        if init_size < 0:
            raise ValueError("init_size (%d) must be positive or zero" % init_size)
        self._max_size = init_size
        self._array = np.empty((self._max_size, self._max_deg + 1), dtype=COEF_DTYPE)

    cdef void init_from_mv(self, COEF_t[:,:] mv, INDEX_t size):
        self._max_size = size
        self._curr_index = size
        self._array = mv

    def __copy__(self):
        cdef Int_Polynomial_Array array = Int_Polynomial_Array(self.get_max_deg(), self._dps)
        array.init_from_mv(self._array, self._max_size)
        return array

    def __deepcopy__(self, memo):
        cdef Int_Polynomial_Array array = Int_Polynomial_Array(self.get_max_deg(), self._dps)
        cdef INDEX_t i
        array.init_empty(self._max_size)
        for i in range(self.get_curr_index()):
            array.append(self.get_poly(i))
        return array


    def __getitem__(self, item):
        cdef Int_Polynomial_Array ret_array
        cdef Int_Polynomial ret_poly
        cdef INDEX_t start, stop, step
        cdef INDEX_t i

        if isinstance(item, slice):
            start = item.start if item.start else 0
            stop = item.stop   if item.stop  else self.get_curr_index()
            step = item.step   if item.step  else 1
            if step <= 0:
                raise IndexError("step must be positive. passed step: %d" % step)
            if start < 0:
                start += self.get_curr_index()
            if stop < 0:
                stop += self.get_curr_index()
                if stop < 0:
                    stop = 0
            if start > stop:
                raise IndexError("start index (%d) exceeds stop index (%d)" % (start, stop))
            if stop > self.get_curr_index():
                stop = self.get_curr_index()
            init_size = (stop - start) // step
            if init_size * step != stop - start:
                init_size += 1
            ret_array = Int_Polynomial_Array(self.get_max_deg(), self._dps)
            if start < stop:
                ret_array.init_from_mv(self._array[start:stop:step, :], init_size)
            return ret_array

        else:
            i = item
            if i >= self.get_len():
                raise IndexError("passed index (%d) exceeds maximum index (%d)" % (i, self.get_len() - 1))
            if i < 0:
                i += self.get_len()
            if i < 0:
                raise IndexError("could not resolve passed index")
            else:
                return self.get_poly(i)

    def __len__(self):
        return self.get_len()

    cdef INDEX_t get_len(self):
        return self._max_size

    def __eq__(self, other):
        return cpb(self.eq(other))

    cdef BOOL_TYPE eq(self, Int_Polynomial_Array other):
        cdef INDEX_t i
        if self.get_len() != other.get_len():
            return FALSE
        if self.get_curr_index() != other.get_curr_index():
            return FALSE
        if self.get_max_deg() != other.get_max_deg():
            return FALSE
        for i in range(self.get_curr_index()):
            if self.get_poly(i) != other.get_poly(i):
                return FALSE
        return TRUE

    def __hash__(self):
        raise TypeError("Int_Polynomial_Array is not a hashable type")

    cdef void set_curr_index(self, INDEX_t curr_index):
        self._curr_index = curr_index

    cdef INDEX_t get_curr_index(self):
        return self._curr_index

    cpdef DEG_t get_max_deg(self):
        return self._max_deg

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void append(self, Int_Polynomial poly) except *:
        cdef DEG_t poly_max_deg = poly.get_max_deg()
        cdef DEG_t i
        if self._curr_index >= self.get_len():
            raise IndexError("This array has reached its max size, %d" % self.get_len())
        if poly_max_deg > self.get_max_deg():
            raise ValueError("passed polynomial max degree (%d) exceeds maximum degree of this array (%d)" % (poly_max_deg, self.get_max_deg()))
        for i in range(self.get_max_deg()+1):
            self._array[self._curr_index, i] = poly.get_coef(i)
        self._curr_index += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void pad(self, INDEX_t pad_size) except *:
        cdef INDEX_t i
        cdef DEG_t j
        cdef COEF_t[:,:] array

        if pad_size < 0:
            raise ValueError("pad_size must be positive, passed pad_size: %d" % pad_size)
        if pad_size == 0:
            return

        array = np.empty((self.get_len() + pad_size, self.get_max_deg() + 1), dtype=COEF_DTYPE)

        for i in range(self._curr_index):
            for j in range(self.get_max_deg() + 1):
                array[i,j] = self._array[i,j]

        self._array = array
        self._max_size += pad_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Int_Polynomial get_poly(self, INDEX_t i):
        if i >= self._curr_index:
            raise IndexError("Max index is %d, passed index is %d" % (self._curr_index-1, i))
        if i < 0:
            raise IndexError("passed index (%d) must be positive" % i)
        return Int_Polynomial(np.asarray(self._array[i, :]), self._dps)

    cpdef cnp.ndarray[COEF_t, ndim = 2] get_ndarray(self):
        return np.asarray(self._array[:self._curr_index, :])