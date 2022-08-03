cimport cython

from intpolynomials cimport *

import warnings

import numpy as np
from mpmath import mpf

COEF_DTYPE = np.int64
DEG_DTYPE = np.int32

def is_int(num):
    return isinstance(num, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64))

NP_INT_DTYPES = (np.int8, np.int16, np.int32, np.int64)
NP_UINT_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)

cdef BOOL_t FALSE = 0
cdef BOOL_t TRUE = 1

cdef ERR_t is_readonly(cnp.ndarray array) except -1:

    if isinstance(array, np.memmap) and array.mode == "r":
        return TRUE

    else:
        return FALSE

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DEG_t calc_deg(const COEF_t[:,:] array, INDEX_t i) except? -1:

    cdef DEG_t j
    cdef DEG_t max_deg = array.shape[1] - 1

    for j in range(max_deg, -1, -1):

        if array[i, j] != 0:
            return j

    return -1

cdef class Int_Polynomial_Array:

    def __init__(self, max_deg):

        self._max_deg = max_deg
        self._is_set = FALSE
        self._curr_index = 0
        self._readonly = FALSE
        self._degs_set = FALSE

    ###############################
    #            ERRORS           #

    cdef inline ERR_t _check_is_set_raise(self) except -1:

        if self._is_set == FALSE:
            raise ValueError(f"`{self.__class__.__name__}` instance is not set.")

        return 0

    cdef inline ERR_t _check_is_not_set_raise(self) except -1:

        if self._is_set == TRUE:
            raise ValueError(f"`{self.__class__.__name__}` instance is already set.")

        return 0

    cdef inline ERR_t _check_readwrite_raise(self) except -1:

        if self._readonly == TRUE:
            raise ValueError(f"`{self.__class__.__name__}` instance is readonly.")

        return 0

    cdef inline ERR_t _check_i_raise(self, INDEX_t i) except -1:

        if i < 0:
            raise ValueError(f"i ({i}) must be non-negative.")

        if i >= self._curr_index:
            raise ValueError(f"i ({i}) must be less than `self._curr_index` ({self._curr_index}).")

        return 0

    cdef inline ERR_t _check_j_raise(self, DEG_t j) except -1:

        if j < 0:
            raise ValueError(f"j ({j}) must be non-negative.")

        if j > self._max_deg:
            raise ValueError(f"j ({j}) must be at most `self._max_deg` ({self._max_deg}).")

        return 0

    cdef inline ERR_t _check_degs_set_raise(self) except -1:

        if self._degs_set == FALSE:
            raise ValueError(f"Must call `c_set_degs` prior to calling this method.")

        return 0

    ###############################
    #      C SETTERS/GETTERS      #

    cdef ERR_t c_set_rw_array(self, COEF_t[:,:] array) except -1:
        """Set the non-`const` array of coefficients.
        
        This method does not calculate polynomial degrees. You must call `c_set_degs` for that.
        
        :param array: The first axis indexes polynomials, the second coefficients.
        """

        self._check_is_not_set_raise()

        if array.shape[1] > self._max_deg + 1:
            raise ValueError(f"Maximum size of `array.shape[1]` is `self._max_deg + 1`.")

        self._readonly = FALSE
        self._rw_array = array
        self._ro_array = self._rw_array
        self._is_set = TRUE
        self._max_len = array.shape[0]
        self._curr_index = self._max_len
        return 0

    cdef ERR_t c_set_ro_array(self, const COEF_t[:,:] array) except -1:
        """Set the non-`const` array of coefficients."""

        self._check_is_not_set_raise()

        if array.shape[1] > self._max_deg + 1:
            raise ValueError(f"Maximum size of `array.shape[1]` is `self._max_deg + 1`.")

        self._readonly = TRUE
        self._ro_array = array
        self._is_set = TRUE
        self._max_len = array.shape[0]
        self._curr_index = self._max_len
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_set_poly(self, INDEX_t i, Int_Polynomial poly) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        poly._check_is_set_raise()
        self._check_readwrite_raise()
        self._check_i_raise(i)

        if self._max_deg < poly._deg:
            raise ValueError(f"`poly` degree is {poly._deg}, but max degree for this `Int_Polynomial_Array` is {self._max_deg}.")

        for j in range(poly._deg + 1):
            self._rw_array[i,j] = poly._ro_coefs[j]

        for j in range(poly._deg + 1, self._max_deg + 1):
            self._rw_array[i,j] = poly._ro_coefs[j]

        if self._degs_set == TRUE:
            self._degs[i] = poly._deg

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_get_poly(self, INDEX_t i, Int_Polynomial poly) except -1:
        """Equivalent to `poly.c_set_array(self, i)`."""

        cdef DEG_t deg

        self._check_is_set_raise()
        poly._check_is_not_set_raise()
        self._check_i_raise(i)

        if self._degs_set == TRUE:
            deg = self._degs[i]

        else:
            deg = calc_deg(self._ro_array, i)

        if deg > poly._max_deg:
            raise ValueError(f"`poly._max_deg` is {poly._max_deg}, but requested polynomial has degree {self._max_deg}.")

        poly.c_set_array(self, i)

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_set_degs(self) except -1:

        cdef INDEX_t i
        cdef DEG_t j

        self._check_is_set_raise()

        self._degs = -np.ones(self._max_len, dtype = DEG_DTYPE)

        for i in range(self._curr_index):
            self._degs[i] = calc_deg(self._ro_array, i)

        return 0

    cpdef ERR_t empty(self, INDEX_t max_len) except -1:

        self._check_is_not_set_raise()

        self._readonly = FALSE
        self._rw_array = np.zeros((max_len, self._max_deg + 1), dtype = COEF_DTYPE)
        self._ro_array = self._rw_array
        self._is_set = TRUE
        self._max_len = max_len
        self._curr_index = 0
        self._degs = -np.ones(max_len, dtype = DEG_DTYPE)

    ###############################
    #     PY SETTERS/GETTERS      #

    def set(self, coefs):

        self._check_is_not_set_raise()

        if isinstance(coefs, (list, tuple)):

            self.set(np.array(coefs, dtype=COEF_DTYPE))

        elif isinstance(coefs, np.ndarray):

            if coefs.dtype in NP_UINT_DTYPES:
                warnings.warn("Int_Polynomial.set : Casting from unsigned int to np.int64 is dangerous.")

            elif coefs.dtype not in NP_INT_DTYPES:
                raise TypeError(f"`{coefs.dtype}` is not a Numpy int type.")

            if len(coefs.shape) != 2:
                raise ValueError("`coefs` must have exactly two dimensions.")

            if coefs.shape[0] == 0 or coefs.shape[1] == 0:
                raise ValueError("`coefs.shape[0]` and `coefs.shape[1]` must both be non-zero.")

            if coefs.shape[1] > self._max_deg + 1:
                raise ValueError("`coefs.shape[1]` must be at most `self._max_deg + 1`.")

            if is_readonly(coefs) == TRUE:
                self.c_set_ro_array(coefs)

            else:
                self.c_set_rw_array(coefs)

            self.c_set_degs()

        else:
            raise TypeError("`coefs` must be a `list`, `tuple`, or `np.ndarray`.")

        return self

    def __setitem__(self, i, poly):
        self.c_set_poly(i, poly)

    def __getitem__(self, i):

        cdef Int_Polynomial poly = Int_Polynomial(self._max_deg)

        self.c_get_poly(i, poly)
        return poly

    ###############################
    #           C SUGAR           #

    cdef ERR_t c_copy(self, Int_Polynomial_Array copy) except -1:

        cdef COEF_t[:,:] _rw_array_copy

        self._check_is_set_raise()
        copy._check_is_not_set_raise()

        if copy._max_deg != self._max_deg:
            raise ValueError("Passed `copy._max_deg` must equal `self._max_deg`.")

        if self._readonly == TRUE:
            copy.c_set_ro_array(self._ro_array)

        else:

            _rw_array_copy = np.empty((self._max_len, self._max_deg + 1), dtype = COEF_DTYPE)
            _rw_array_copy[:,:] = self._rw_array
            copy.c_set_rw_array(_rw_array_copy)

        if self._degs_set == TRUE:

            copy._degs = np.empty(self._max_len, dtype = COEF_DTYPE)
            copy._degs[:] = self._degs
            copy._degs_set = TRUE

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef COEF_t max_abs_coef(self) except -1:

        cdef INDEX_t i
        cdef DEG_t j
        cdef COEF_t curr_max = 0
        cdef COEF_t c

        self._check_is_set_raise()

        if self._deg == -1:
            return 0

        for i in range(self._curr_index):

            for j in range(self._max_deg + 1):

                c = self._ro_coefs[i]

                if c < 0 and -c > curr_max:
                    curr_max = -c

                elif c >= 0 and c > curr_max:
                    curr_max = c

        return curr_max

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_eq(self, Int_Polynomial_Array other) except -1:

        cdef INDEX_t i
        cdef DEG_t j

        self._check_is_set_raise()
        other._check_is_set_raise()

        if type(self) != type(other):
            return FALSE

        if self._curr_index != other._curr_index or self._max_deg != other._max_deg:
            return FALSE

        for i in range(self._curr_index):

            for j in range(self._max_deg + 1):

                if self._ro_array[i, j] != other._ro_array[i, j]:
                    return FALSE

        else:
            return TRUE

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ERR_t append(self, Int_Polynomial poly) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        poly._check_is_set_raise()
        self._check_readwrite_raise()

        if self._curr_index >= self._max_len:
            raise ValueError("This `Int_Polynomial_Array` is full.")

        for j in range(poly._deg + 1):
            self._rw_array[self._curr_index, j] = poly._ro_coefs[j]

        self._curr_index += 1
        return 0

    ###############################
    #          PY SUGAR           #

    def get_ndarray(self):

        self._check_is_set_raise()

        if self._readonly == TRUE:
            return np.asarray(self._ro_array)

        else:
            return np.asarray(self._rw_array)

    def __copy__(self):

        cdef Int_Polynomial_Array copy = Int_Polynomial_Array(self._max_deg)
        self.c_copy(copy)
        return copy

    def __deepcopy__(self, memo):
        return self.__copy__()

    def max_deg(self):
        return self._max_deg

    def degs(self):

        self._check_is_set_raise()

        if self._degs_set == FALSE:
            self.c_set_degs()

        return np.asarray(self._degs)[:self._curr_index]

    def __str__(self):

        self._check_is_set_raise()

        return str(self.get_ndarray())

    def __repr__(self):
        return str(self)

    def __ne__(self, other):
        return not(self == other)

    def __eq__(self, other):

        return self.c_eq(other) == TRUE

    def __len__(self):
        return self._curr_index

cdef class Int_Polynomial(Int_Polynomial_Array):


    ###############################
    #      C SETTERS/GETTERS      #

    cdef ERR_t c_set_array(self, Int_Polynomial_Array array, INDEX_t index) except -1:

        self._check_is_not_set_raise()

        if self._max_deg != array._max_deg:
            raise ValueError("`Int_Polynomial` max deg must equal `array` max deg.")

        if array._readonly == TRUE:

            Int_Polynomial_Array.c_set_ro_array(self, array._ro_array)
            self._ro_coefs = self._ro_array[index, :]

        else:

            Int_Polynomial_Array.c_set_rw_array(self, array._rw_array)
            self._rw_coefs = self._rw_array[index, :]
            self._ro_coefs = self._rw_coefs

        self._index = index

        if self._degs_set == TRUE:
            self._deg = self._degs[self._index]

        else:
            self._deg = calc_deg(self._ro_array, self._index)

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_set_coef(self, DEG_t j, COEF_t c) except -1:

        self._check_is_set_raise()
        self._check_readwrite_raise()
        self._check_j_raise(j)

        self._rw_coefs[j] = c

        if c != 0 and j > self._deg:
            self._deg = j

        elif c == 0 and j == self._deg:
            self._deg = calc_deg(self._ro_array, self._index)

        if self._degs_set == TRUE:
            self._degs[self._index] = self._deg

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COEF_t c_get_coef(self, DEG_t j) except? -1:

        self._check_is_set_raise()
        self._check_j_raise(j)

        return self._ro_coefs[j]

    cpdef ERR_t zero_poly(self) except -1:

        cdef COEF_t[:,:] array

        self._check_is_not_set_raise()

        array = np.zeros((1, self._max_deg + 1), dtype = COEF_DTYPE)
        self.c_set_array(array, 0)
        self._deg = -1
        return 0

    ###############################
    #     PY SETTERS/GETTERS      #

    def set(self, coefs):

        cdef Int_Polynomial_Array array

        self._check_is_not_set_raise()

        if isinstance(coefs, (list, tuple)):
            self.set(np.array(coefs, dtype=COEF_DTYPE))

        elif isinstance(coefs, np.ndarray):

            if coefs.dtype in NP_UINT_DTYPES:
                warnings.warn("Int_Polynomial.set : Casting from unsigned int to np.int64 is dangerous.")

            elif coefs.dtype not in NP_INT_DTYPES:
                raise TypeError(f"`{coefs.dtype}` is not a Numpy int type.")

            if len(coefs.shape) != 1:
                raise ValueError("`coefs` must have exactly one dimension.")

            if coefs.shape[0] == 0:
                raise ValueError("`coefs.shape[0]` must be non-zero.")

            if coefs.shape[0] > self._max_deg + 1:
                raise ValueError("`coefs.shape[0]` must be at most `self._max_deg + 1`.")

            array = Int_Polynomial_Array(coefs.shape[0] - 1)

            if is_readonly(coefs) == TRUE:
                array.c_set_ro_array(coefs[np.newaxis, :])

            else:
                array.c_set_rw_array(coefs[np.newaxis, :])

            self.c_set_array(array, 0)

        else:
            raise TypeError("`coefs` must be a `list`, `tuple`, or `np.ndarray`.")

        return self

    def __setitem__(self, j, coef):
        self.c_set_coef(j, coef)

    def __getitem__(self, j):
        return self.c_get_coef(j)

    ###############################
    #           C SUGAR           #

    cdef ERR_t c_copy(self, Int_Polynomial_Array copy) except -1:

        cdef COEF_t[:,:] array
        cdef DEG_t j

        self._check_is_set_raise()
        copy._check_is_not_set_raise()

        if not isinstance(copy, Int_Polynomial):
            raise TypeError("`copy` must be of type `Int_Polynomial`.")

        array = np.empty((1, self._max_deg + 1), dtype = COEF_DTYPE)

        for j in range(self._max_deg + 1):
            array[0, j] = self._ro_coefs[j]

        copy.c_set_array(array, 0)
        return 0

    cdef ERR_t c_eval(self, MPF_t x, BOOL_t calc_deriv) except -1:

        cdef:
            MPF_t p
            MPF_t q
            COEF_t c
            DEG_t j

        self._check_is_set_raise()

        if self._deg < 0:

            p = mpf(0.0)
            q = mpf(0.0)

        else:

            p = mpf(self._ro_coefs[self._deg])
            q = mpf(0)

            for j in range(self._deg - 1, -1, -1):

                c  = self._ro_coefs[j]

                if calc_deriv == TRUE:
                    q = p + x * q

                p = x * p + c

        self.last_eval = p
        self.last_deriv = q

        return 0

    cdef ERR_t c_eq(self, Int_Polynomial_Array other) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        other._check_is_set_raise()

        if type(self) != type(other) or self._deg != (<Int_Polynomial> other)._deg:
            return FALSE

        for j in range(self._deg + 1):

            if self._ro_coefs[j] != (<Int_Polynomial> other)._ro_coefs[j]:
                return FALSE

        else:
            return TRUE


    ###############################
    #          PY SUGAR           #

    def get_ndarray(self):

        self._check_is_set_raise()

        if self._readonly == TRUE:
            return np.asarray(self._ro_coefs)

        else:
            return np.asarray(self._rw_coefs)

    def __copy__(self):

        cdef Int_Polynomial copy = Int_Polynomial(self._max_deg)
        self.c_copy(copy)
        return copy

    def deg(self):

        self._check_is_set_raise()

        return self._deg

    def __call__(self, x, calc_deriv = False):

        self.c_eval(x, TRUE if calc_deriv else FALSE)

        if calc_deriv:
            return self.last_eval, self.last_deriv

        else:
            return self.last_eval

    ###############################
    #         INVALIDATED         #

    cdef ERR_t c_set_rw_array(self, COEF_t[:,:] array) except -1:
        raise NotImplementedError("Cannot call `c_set_rw_array` on `Int_Polynomial`.")

    cdef ERR_t c_set_ro_array(self, const COEF_t[:,:] array) except -1:
        raise NotImplementedError("Cannot call `c_set_ro_array` on `Int_Polynomial`.")

    cdef ERR_t c_set_poly(self, INDEX_t i, Int_Polynomial poly) except -1:
        raise NotImplementedError("Cannot call `c_set_poly` on `Int_Polynomial`.")

    cdef ERR_t c_get_poly(self, INDEX_t i, Int_Polynomial poly) except -1:
        raise NotImplementedError("Cannot call `c_get_poly` on `Int_Polynomial`.")

    cdef ERR_t c_set_degs(self) except -1:
        raise NotImplementedError("Cannot call `c_set_degs` on `Int_Polynomial`.")

    cpdef ERR_t append(self, Int_Polynomial poly) except -1:
        raise NotImplementedError("Cannot call `append` on `Int_Polynomial`.")

    def __len__(self):
        raise NotImplementedError("Cannot call `len` on `Int_Polynomial`.")