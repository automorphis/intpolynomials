cimport cython

from intpolynomials cimport *

import warnings
import math

import numpy as np
import mpmath
import xxhash

try:
    from sage.all import PolynomialRing, ComplexField, ZZ

except ModuleNotFoundError:
    PolynomialRing = ComplexField = ZZ = None

IF UNAME_SYSNAME == "Windows":

    COEF_DTYPE = np.int64
    DEG_DTYPE = np.int32

ELIF UNAME_SYSNAME == "Linux":

    COEF_DTYPE = np.longlong
    DEG_DTYPE = int

def is_int(num):
    return isinstance(num, (int,) + NP_INT_DTYPES + NP_UINT_DTYPES)

NP_INT_DTYPES = (np.int8, np.int16, np.int32, np.int64)
NP_UINT_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)

cdef BOOL_t FALSE = 0
cdef BOOL_t TRUE = 1

cdef COEF_t gcd(COEF_t a, COEF_t b) except -1:

    cdef COEF_t q
    cdef COEF_t temp

    if a == 0 and b == 0:
        raise ZeroDivisionError

    if a < 0:
        a = -a

    if b < 0:
        b = -b

    if a == 0:
        return b

    if b == 0:
        return a

    if a > b:

        temp = a
        a = b
        b = temp

    r = b % a

    while r != 0:

        b = a
        a = r
        r = b % a

    return a

cdef COEF_t mv_gcd(const COEF_t[:] coefs) except -1:

    cdef DEG_t j
    cdef COEF_t g, c

    if coefs.shape[0] == 0:
        raise ValueError("coefs cannot be empty")

    g = coefs[0]

    if g < 0:
        g = -g

    for j in range(1, coefs.shape[0]):

        if g == 1:
            return g

        c = coefs[j]

        if c != 0:
            g = gcd(g, c)

    if g == 0:
        raise ZeroDivisionError

    return g


cdef COEF_t lcm(COEF_t a, COEF_t b) except -1:

    if a == 0 or b == 0:
        return 0

    if a < 0:
        a = -a

    if b < 0:
        b = -b

    return a * b // gcd(a, b)

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

cdef class IntPolynomialArray:

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

        if array.shape[1] != self._max_deg + 1:
            raise ValueError(f"`array.shape[1]` must be equal to 1 plus the maximum degree of this array.")

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
            raise ValueError(f"`array.shape[1]` must be equal to 1 plus the maximum degree of this array.")

        self._readonly = TRUE
        self._ro_array = array
        self._is_set = TRUE
        self._max_len = array.shape[0]
        self._curr_index = self._max_len
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_set_poly(self, INDEX_t i, IntPolynomial poly) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        poly._check_is_set_raise()
        self._check_readwrite_raise()
        self._check_i_raise(i)

        if self._max_deg < poly._deg:
            raise ValueError(f"`poly` degree is {poly._deg}, but max degree for this `IntPolynomialArray` is {self._max_deg}.")

        for j in range(poly._deg + 1):
            self._rw_array[i, j] = poly._ro_coefs[j]

        for j in range(poly._deg + 1, self._max_deg + 1):
            self._rw_array[i, j] = 0

        if self._degs_set == TRUE:
            self._degs[i] = poly._deg

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_get_poly(self, INDEX_t i, IntPolynomial poly) except -1:
        """Equivalent to `poly.c_set_array(self, i)`."""

        cdef DEG_t deg

        self._check_is_set_raise()
        poly._check_is_not_set_raise()
        self._check_i_raise(i)

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

    cpdef ERR_t zeros(self, INDEX_t max_len) except -1:

        self.empty(max_len)
        self._curr_index = max_len

        return 0

    cpdef ERR_t ones(self, INDEX_t max_len) except -1:

        cdef INDEX_t i

        self.zeros(max_len)

        for i in range(self._max_len):
            self._rw_array[i, 0] = 1

        return 0

    cpdef ERR_t clear(self) except -1:

        self._check_is_set_raise()

        self._curr_index = 0

        return 0

    cpdef ERR_t set_max_deg(self, DEG_t max_deg) except -1:

        self._check_is_not_set_raise()

        self._max_deg = max_deg

        return 0

    cpdef ERR_t get_max_deg(self) except? -1:

        return self._max_deg

    ###############################
    #     PY SETTERS/GETTERS      #

    def set(self, coefs):

        self._check_is_not_set_raise()

        if isinstance(coefs, (list, tuple)):

            self.set(np.array(coefs, dtype=COEF_DTYPE))

        elif isinstance(coefs, np.ndarray):

            if coefs.dtype in NP_UINT_DTYPES:
                warnings.warn("IntPolynomial.set : Casting from unsigned int to np.int64 is dangerous.")

            elif coefs.dtype not in NP_INT_DTYPES:
                raise TypeError(f"`{coefs.dtype}` is not a Numpy int type.")

            if len(coefs.shape) != 2:
                raise ValueError("`coefs` must have exactly two dimensions.")

            if coefs.shape[0] == 0 or coefs.shape[1] == 0:
                raise ValueError("`coefs.shape[0]` and `coefs.shape[1]` must both be non-zero.")

            if coefs.shape[1] > self._max_deg + 1:
                raise ValueError(f"`array.shape[1]` must be equal to the maximum degree of this array.")

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

        cdef IntPolynomial poly = IntPolynomial(self._max_deg)

        self.c_get_poly(i, poly)
        return poly

    ###############################
    #           C SUGAR           #

    cdef ERR_t c_copy(self, IntPolynomialArray copy) except -1:

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

        for i in range(self._curr_index):

            for j in range(self._max_deg + 1):

                c = self._ro_array[i, j]

                if c < 0 and -c > curr_max:
                    curr_max = -c

                elif c >= 0 and c > curr_max:
                    curr_max = c

        return curr_max

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_eq(self, IntPolynomialArray other) except -1:

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
    cpdef ERR_t append(self, IntPolynomial poly) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        poly._check_is_set_raise()
        self._check_readwrite_raise()

        if self._curr_index >= self._max_len:
            raise ValueError("This `IntPolynomialArray` is full.")

        # print(np.asarray(self._ro_array))

        for j in range(poly._max_deg + 1):
            # print(poly._ro_coefs[j])
            # print(self._rw_array[self._curr_index, j])
            self._rw_array[self._curr_index, j] = poly._ro_coefs[j]
            # print(poly._ro_coefs[j])
            # print(self._rw_array[self._curr_index, j])

        self._curr_index += 1
        return 0

    cdef INDEX_t c_len(self) except -1:

        self._check_is_set_raise()

        return self._curr_index

    ###############################
    #          PY SUGAR           #

    def __iter__(self):
        return IntPolynomialArray_Iter(self)

    def get_ndarray(self):

        self._check_is_set_raise()

        if self._readonly == TRUE:
            return np.asarray(self._ro_array[ : self._curr_index, :])

        else:
            return np.asarray(self._rw_array[ : self._curr_index, :])

    def __copy__(self):

        cdef IntPolynomialArray copy = IntPolynomialArray(self._max_deg)
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

cdef class IntPolynomialArray_Iter:

    def __init__(self, array):

        self.array = array
        self.i = 0

    def __next__(self):

        cdef IntPolynomial poly

        if self.i >= self.array._curr_index:
            raise StopIteration

        else:

            poly = IntPolynomial(self.array._max_deg)
            self.array.c_get_poly(self.i, poly)
            self.i += 1
            return poly


cdef class IntPolynomial(IntPolynomialArray):

    def __init__(self, max_deg):

        self._hashed = FALSE
        super().__init__(max_deg)

    ###############################
    #      C SETTERS/GETTERS      #

    cdef ERR_t c_set_array(self, IntPolynomialArray array, INDEX_t index) except -1:

        self._check_is_not_set_raise()

        if self._max_deg != array._max_deg:
            raise ValueError("`IntPolynomial` max deg must equal `array` max deg.")

        if array._readonly == TRUE:

            IntPolynomialArray.c_set_ro_array(self, array._ro_array)
            self._ro_coefs = self._ro_array[index, :]

        else:

            IntPolynomialArray.c_set_rw_array(self, array._rw_array)
            self._rw_coefs = self._rw_array[index, :]
            self._ro_coefs = self._rw_coefs

        self._index = index

        if self._degs_set == TRUE:
            self._deg = self._degs[self._index]

        else:
            self._deg = calc_deg(self._ro_array, self._index)

        self._hashed = FALSE
        self._lcd = 1

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

        self._hashed = FALSE

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COEF_t c_get_coef(self, DEG_t j) except? -1:

        self._check_is_set_raise()
        self._check_j_raise(j)

        return self._ro_coefs[j]

    cpdef ERR_t zero_poly(self) except -1:

        cdef COEF_t[:,:] array
        cdef IntPolynomialArray iparray

        self._check_is_not_set_raise()

        array = np.zeros((1, self._max_deg + 1), dtype = COEF_DTYPE)
        iparray = IntPolynomialArray(self._max_deg)
        iparray.c_set_rw_array(array)
        self.c_set_array(iparray, 0)
        self._deg = -1
        return 0

    cpdef ERR_t set_lcd(self, COEF_t lcd) except -1:

        self._check_is_set_raise()

        if lcd <= 0:
            raise ValueError("`lcd` must be positive.")

        self._lcd = lcd

    cpdef ERR_t get_lcd(self) except -1:

        self._check_is_set_raise()

        return self._lcd

    ###############################
    #     PY SETTERS/GETTERS      #

    def set(self, coefs):

        cdef IntPolynomialArray array

        self._check_is_not_set_raise()

        if isinstance(coefs, (list, tuple)):
            self.set(np.array(coefs, dtype=COEF_DTYPE))

        elif isinstance(coefs, np.ndarray):

            if coefs.dtype in NP_UINT_DTYPES:
                warnings.warn("IntPolynomial.set : Casting from unsigned int to np.int64 is dangerous.")

            elif coefs.dtype not in NP_INT_DTYPES:
                raise TypeError(f"`{coefs.dtype}` is not a Numpy int type.")

            if len(coefs.shape) != 1:
                raise ValueError("`coefs` must have exactly one dimension.")

            if coefs.shape[0] == 0:
                raise ValueError("`coefs.shape[0]` must be non-zero.")

            if coefs.shape[0] > self._max_deg + 1:
                raise ValueError("`coefs.shape[0]` must be at most `self._max_deg + 1`.")

            array = IntPolynomialArray(coefs.shape[0] - 1)

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

    cdef ERR_t c_copy(self, IntPolynomialArray copy) except -1:

        cdef DEG_t j
        cdef IntPolynomialArray array

        self._check_is_set_raise()
        copy._check_is_not_set_raise()

        if not isinstance(copy, IntPolynomial):
            raise TypeError("`copy` must be of type `IntPolynomial`.")

        if copy._max_deg != self._max_deg:
            raise ValueError("`copy` has the wrong max degree.")

        array = IntPolynomialArray(self._max_deg)
        array.zeros(1)

        for j in range(self._deg + 1):
            array._rw_array[0, j] = self._ro_coefs[j]

        (<IntPolynomial> copy).c_set_array(array, 0)
        (<IntPolynomial> copy).set_lcd(self._lcd)
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_eval(self, MPF_t x, BOOL_t calc_deriv) except -1:

        cdef:
            MPF_t p
            MPF_t q
            COEF_t c
            DEG_t j

        self._check_is_set_raise()

        if self._deg < 0:

            p = mpmath.mpf(0.0)
            q = mpmath.mpf(0.0)

        else:

            p = mpmath.mpf(self._ro_coefs[self._deg])
            q = mpmath.mpf(0)

            for j in range(self._deg - 1, -1, -1):

                c  = self._ro_coefs[j]

                if calc_deriv == TRUE:
                    q = p + x * q

                p = x * p + c

        self.last_eval = p / self._lcd
        self.last_deriv = q / self._lcd

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_eq(self, IntPolynomialArray other) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        other._check_is_set_raise()

        if type(self) != type(other) or self._deg != (<IntPolynomial> other)._deg:
            return FALSE

        if self._lcd != (<IntPolynomial> other)._lcd:
            return FALSE

        for j in range(self._deg + 1):

            if self._ro_coefs[j] != (<IntPolynomial> other)._ro_coefs[j]:
                return FALSE

        else:
            return TRUE

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_deriv(self, IntPolynomial deriv) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        deriv._check_is_not_set_raise()

        deriv.zero_poly()

        if self._deg == -1:
            return 0

        for j in range(1, self._deg + 1):
            deriv._rw_coefs[j - 1] = j * self._ro_coefs[j]

        deriv._deg = self._deg - 1

        return 0

    cpdef COEF_t content(self) except -1:

        cdef COEF_t cont, coef
        cdef DEG_t j

        self._check_is_set_raise()

        if self._deg == -1:
            raise ValueError("cannot calculate content of zero polynomial.")

        return mv_gcd(self._ro_coefs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef COEF_t sum_abs_coef(self) except -1:

        cdef DEG_t j
        cdef COEF_t ret, c

        self._check_is_set_raise()

        ret = 0

        for j in range(self._deg + 1):

            c = self._ro_coefs[j]

            if c < 0:
                ret += -c

            else:
                ret += c

        return ret

        # cpdef ERR_t normalize(self) except -1:
    #
    #     cdef COEF_t cont, div
    #     cdef DEG_t j
    #
    #     self._check_is_set_raise()
    #     self._check_readwrite_raise()
    #
    #     cont = self.content()
    #     div = gcd(cont, self._lcd)
    #     self._lcd = self._lcd // div
    #
    #     if self._lcd < 0:
    #
    #         self._lcd = -self._lcd
    #         div = -div
    #
    #     for j in range(self._deg + 1):
    #         self._rw_coefs[j] = self._ro_coefs[j] // div

    # cdef ERR_t c_roots(self, object roots_and_roots_abs_and_mult) except -1:



        # cdef IntPolynomialArray fact
        # cdef IntPolynomial sqf
        # cdef COEF_t leading, scale, discr
        # cdef INDEX_t step, restart, i
        # cdef DEG_t j, j0
        # cdef MPF_t j0_root, j0_root_abs
        # cdef MPF_t max_error, x, x_abs, running_error, last_running_error, denom, tol, real_comp, imag_comp
        # cdef DEG_t[:] indices_pop
        # cdef cnp.ndarray[cnp.float_t, ndim = 2] flattened_coords
        # cdef cnp.ndarray[DEG_t, ndim = 1] rand_indices
        # cdef cnp.float_t[:, :] initial_roots
        # cdef float norm
        # cdef BOOL_t next_restart, neg_discr
        # cdef object these_roots, these_roots_and_roots_abs_and_mult
        #
        # self._check_is_set_raise()
        #
        # if not isinstance(roots_and_roots_abs_and_mult, list):
        #     raise TypeError("`roots_and_roots_abs_and_mult` must be of type `list`.")
        #
        # if len(roots_and_roots_abs_and_mult) != 0:
        #     raise ValueError("`roots_and_roots_abs_and_mult` must be an empty `list`.")
        #
        # if self._deg <= 0:
        #     raise ValueError("Can only calculate roots of polynomials with degree at least 1.")
        #
        # if max_steps <= 0:
        #     raise ValueError("`max_steps` must be positive.")
        #
        # if max_restarts <= 0:
        #     raise ValueError("`max_restarts` must be positive.")
        #
        # fact = IntPolynomialArray(self._max_deg)
        # self.c_sqfree_fact(fact)
        #
        # with extradps(self._deg):
        #
        #     tol = power(10, -mp.dps)
        #
        #     for i in range(1, fact.c_len()):
        #
        #         sqf = IntPolynomial(fact._max_deg)
        #         fact.c_get_poly(i, sqf)
        #
        #         if sqf._deg == 0:
        #             continue
        #
        #         elif sqf._deg == 1:
        #
        #             j0_root = mpf(-sqf.c_get_coef(0)) / mpf(sqf.c_get_coef(1))
        #             j0_root_abs = fabs(j0_root)
        #             roots_and_roots_abs_and_mult.extend([(j0_root, j0_root_abs, i)])
        #             continue
        #
        #         elif sqf._deg == 2:
        #
        #             discr = sqf.c_get_coef(1) ** 2 - 4 * sqf.c_get_coef(0) * sqf.c_get_coef(2)
        #
        #             if discr < 0:
        #
        #                 neg_discr = TRUE
        #                 discr = -discr
        #
        #             else:
        #                 neg_discr = FALSE
        #
        #             real_comp = -mpf(sqf.c_get_coef(1)) / mpf(2 * sqf.c_get_coef(2))
        #             imag_comp = mpf(discr) ** 0.5 / mpf(2 * sqf.c_get_coef(2))
        #
        #             if neg_discr == TRUE:
        #
        #                 j0_root = mpc(real_comp, imag_comp)
        #                 j0_root_abs = fabs(j0_root)
        #                 roots_and_roots_abs_and_mult.extend([(j0_root, j0_root_abs, i), (conj(j0_root), j0_root_abs, i)])
        #
        #             else:
        #
        #                 j0_root = real_comp + imag_comp
        #                 j0_root_abs = fabs(j0_root)
        #                 roots_and_roots_abs_and_mult.extend([(j0_root, j0_root_abs, i)])
        #                 j0_root = real_comp - imag_comp
        #                 j0_root_abs = fabs(j0_root)
        #                 roots_and_roots_abs_and_mult.extend([(j0_root, j0_root_abs, i)])
        #
        #             continue
        #
        #         leading = sqf.c_get_coef(sqf._deg)
        #         these_roots_and_roots_abs_and_mult = []
        #         indices_pop = np.arange(sqf._deg ** 2, dtype = DEG_DTYPE)
        #         scale = 1 + sqf.max_abs_coef()
        #         norm = 2 * scale / sqf._deg
        #         flattened_coords = -scale + norm * np.column_stack((
        #             np.repeat(np.arange(sqf._deg, dtype = float), sqf._deg),
        #             np.tile(np.arange(sqf._deg, dtype = float), sqf._deg)
        #         ))
        #
        #         for restart in range(max_restarts):
        #
        #             next_restart = FALSE
        #             running_error = mpf('0.0')
        #
        #             # initialize roots
        #             rand_indices = np.random.choice(indices_pop, sqf._deg, replace = False)
        #             initial_roots = flattened_coords[rand_indices, :] + 0.5 * norm * np.random.rand(sqf._deg, 2)
        #
        #             these_roots = [None] * sqf._deg
        #
        #             for j0 in range(sqf._deg):
        #                 these_roots[j0] = mpc(initial_roots[j0, 0], initial_roots[j0, 1])
        #
        #             for step in range(max_steps):
        #
        #                 for j0 in range(sqf._deg):
        #
        #                     denom = mpf("1.0")
        #                     j0_root = these_roots[j0]
        #
        #                     for j in range(sqf._deg):
        #
        #                         if j != j0:
        #                             denom *= j0_root - these_roots[j]
        #
        #                     sqf.c_eval(j0_root, FALSE)
        #                     denom *= leading
        #
        #                     try:
        #                         x = sqf.last_eval / denom
        #
        #                     except ZeroDivisionError:
        #                         pass
        #
        #                     else:
        #
        #                         x_abs = fabs(x)
        #
        #                         if j0 == 0 or max_error < x_abs:
        #                             max_error = x_abs
        #
        #                         these_roots[j0] = j0_root - x
        #
        #                 if max_error < tol:
        #                     break
        #
        #                 # running_error += max_error
        #                 #
        #                 # error_check_index = step // error_check_period
        #                 #
        #                 # if error_check_index * error_check_period + error_check_period - 1 == step:
        #                 #
        #                 #     # if error_check_index > 0 and last_running_error < running_error:
        #                 #     #
        #                 #     #     next_restart = TRUE
        #                 #     #     print(f"{str(self)}, larger running error, {restart}, {step}, {roots}, {}")
        #                 #     #     break
        #                 #
        #                 #     last_running_error = running_error
        #                 #     running_error = mpf("0.0")
        #
        #             else:
        #                 next_restart = TRUE
        #
        #             if next_restart == TRUE:
        #                 continue
        #
        #             these_roots_and_roots_abs_and_mult.extend([None] * sqf._deg)
        #
        #             for j0 in range(sqf._deg):
        #                 # nudge small roots
        #                 j0_root = these_roots[j0]
        #                 j0_root_abs = fabs(j0_root)
        #
        #                 if j0_root_abs < tol:
        #
        #                     j0_root = mpf("0.0")
        #                     j0_root_abs = mpf("0.0")
        #
        #                 elif fabs(im(j0_root)) < tol:
        #
        #                     j0_root = j0_root.real
        #                     j0_root_abs = fabs(these_roots[j0])
        #
        #                 elif fabs(re(j0_root)) < tol:
        #
        #                     j0_root = j0_root.imag * 1j
        #                     j0_root_abs = fabs(these_roots[j0])
        #
        #                 these_roots_and_roots_abs_and_mult[j0] = (j0_root, j0_root_abs, i)
        #
        #             break # restart loop
        #
        #         else:
        #             raise mp.NoConvergence(
        #                 f"Root finding algorithm did not converge in {max_steps} steps and {max_restarts} restarts."
        #             )
        #
        #         roots_and_roots_abs_and_mult.extend(these_roots_and_roots_abs_and_mult)
        #
        #     roots_and_roots_abs_and_mult.sort(key = lambda t: -t[1])
        #     return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_divide(self, IntPolynomial denom, IntPolynomial quo, IntPolynomial rem) except -1:

        cdef cnp.ndarray[COEF_t, ndim = 2] _rem, _quo
        cdef cnp.ndarray[COEF_t, ndim = 1] _denom, _quo_bottoms, _quo_tops
        cdef DEG_t j, quo_deg, rem_deg
        cdef COEF_t a, b, ap, bp, g, cum_quo_bottom, quo_lcd, quo_gcd, rem_lcd, rem_gcd
        cdef IntPolynomialArray quo_int_poly_array
        cdef IntPolynomialArray rem_int_poly_array

        self._check_is_set_raise()
        denom._check_is_set_raise()
        quo._check_is_not_set_raise()
        rem._check_is_not_set_raise()

        if denom._deg == -1:
            raise ZeroDivisionError

        if self._deg >= denom._deg:
            quo_deg = self._deg - denom._deg

        else:
            quo_deg = -1

        if quo._max_deg != quo_deg:
            raise ValueError(f"`quo` max degree is wrong. Expected {quo_deg} but got {quo._max_deg}.")

        if quo_deg == -1:

            quo.zero_poly()
            rem.set_max_deg(self._max_deg)
            self.c_copy(rem)
            return 0

        _denom = np.empty(denom._deg + 1, dtype = COEF_DTYPE)
        _denom[:] = denom._ro_coefs
        a = _denom[denom._deg]

        _rem = np.empty((1, self._deg + 1), dtype = COEF_DTYPE)
        _rem[0, :] = self._ro_coefs[ : self._deg + 1]

        _quo = np.empty((1, quo_deg + 1), dtype = COEF_DTYPE)
        _quo_bottoms = np.empty(quo_deg + 1, dtype = COEF_DTYPE)
        _quo_tops = np.empty(quo_deg + 1, dtype = COEF_DTYPE)
        j = quo_deg
        cum_quo_bottom = self._lcd

        while j >= 0:

            b = _rem[0, j + denom._deg]
            g = gcd(a, b)
            bp = b // g
            ap = a // g
            cum_quo_bottom *= ap
            _quo_bottoms[j] = cum_quo_bottom
            _quo_tops[j] = bp
            _rem[0,  : j + denom._deg] *= ap
            _rem[0, j : j + denom._deg] -= bp * _denom[ : denom._deg]
            _rem[0, j + denom._deg] = 0
            j -= 1

            while j >= 0 and _rem[0, j + denom._deg] == 0:

                _quo_tops[j] = 0
                _quo_bottoms[j] = cum_quo_bottom
                j -= 1

        _quo_tops *= denom._lcd
        quo_lcd = _quo_bottoms[0]

        for j in range(quo_deg + 1):
            _quo_tops[j] *= quo_lcd // _quo_bottoms[j]

        quo_gcd = gcd(quo_lcd, mv_gcd(_quo_tops))
        quo_lcd //= quo_gcd
        _quo[0, :] = _quo_tops // quo_gcd

        for j in range(denom._deg - 1, -1, -1):

            if _rem[0, j] != 0:

                rem_deg = j
                break

        else:
            rem_deg = -1

        if rem_deg >= 0:

            rem_lcd = cum_quo_bottom
            rem_gcd = gcd(rem_lcd, mv_gcd(_rem[0, : rem_deg + 1]))
            rem_lcd //= rem_gcd
            _rem //= rem_gcd

        else:
            rem_lcd = 1

        if quo_lcd < 0 <= quo_deg:

            _quo *= -1
            quo_lcd *= -1

        if rem_lcd < 0 <= rem_deg:

            _rem *= -1
            rem_lcd *= -1

        quo_int_poly_array = IntPolynomialArray(quo_deg)
        quo_int_poly_array.c_set_rw_array(_quo)
        quo.c_set_array(quo_int_poly_array, 0)
        quo._lcd = quo_lcd

        rem_int_poly_array = IntPolynomialArray(rem_deg)
        rem_int_poly_array.c_set_rw_array(_rem[:, :rem_deg + 1])
        rem.set_max_deg(rem_deg )
        rem.c_set_array(rem_int_poly_array, 0)
        rem._lcd = rem_lcd

        return 0

    cdef ERR_t c_divide_int(self, COEF_t c) except -1:

        cdef DEG_t j

        self._check_is_set_raise()

        for j in range(self._deg + 1):
            self._rw_coefs[j] = self._ro_coefs[j] // c

        return 0

    cdef ERR_t c_subtract(self, IntPolynomial subtra, IntPolynomial diff) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        subtra._check_is_set_raise()
        diff._check_is_not_set_raise()

        if (
            (self._max_deg < subtra._max_deg and diff._max_deg != subtra._max_deg) or
            (subtra._max_deg <= self._max_deg and diff._max_deg != self._max_deg)
        ):
            raise ValueError("`diff` has the wrong max degree")

        diff.zero_poly()

        if self._deg < subtra._deg:

            for j in range(self._deg + 1):
                diff._rw_coefs[j] = self._ro_coefs[j] - subtra._ro_coefs[j]

            for j in range(self._deg + 1, subtra._deg + 1):
                diff._rw_coefs[j] = -subtra._ro_coefs[j]

        else:

            for j in range(subtra._deg + 1):
                diff._rw_coefs[j] = self._ro_coefs[j] - subtra._ro_coefs[j]

            for j in range(subtra._deg + 1, self._deg + 1):
                diff._rw_coefs[j] = self._ro_coefs[j]

        diff._deg = calc_deg(diff._ro_array, 0)

        return 0

    cdef ERR_t c_add(self, IntPolynomial addend, IntPolynomial _sum) except -1:

        cdef DEG_t j

        self._check_is_set_raise()
        addend._check_is_set_raise()
        _sum._check_is_not_set_raise()

        if (
            (self._max_deg < addend._max_deg and _sum._max_deg != addend._max_deg) or
            (addend._max_deg <= self._max_deg and _sum._max_deg != self._max_deg)
        ):
            raise ValueError("`diff` has the wrong max degree")

        _sum.zero_poly()

        if self._deg < addend._deg:

            for j in range(self._deg + 1):
                _sum._rw_coefs[j] = self._ro_coefs[j] + addend._ro_coefs[j]

            for j in range(self._deg + 1, addend._deg + 1):
                _sum._rw_coefs[j] = addend._ro_coefs[j]

        else:

            for j in range(addend._deg + 1):
                _sum._rw_coefs[j] = self._ro_coefs[j] + addend._ro_coefs[j]

            for j in range(addend._deg + 1, self._deg + 1):
                _sum._rw_coefs[j] = self._ro_coefs[j]

        _sum._deg = calc_deg(_sum._ro_array, 0)

        return 0

    cdef ERR_t c_gcd(self, IntPolynomial other, IntPolynomial g) except -1:

        cdef DEG_t j
        cdef IntPolynomial a, b, r, q
        cdef COEF_t g_cont, sign

        self._check_is_set_raise()
        other._check_is_set_raise()
        g._check_is_not_set_raise()

        if self._deg == -1 and other._deg == -1:
            raise ZeroDivisionError

        if self._deg == -1 or other._deg == -1:

            if self._deg == -1:

                g.set_max_deg(other._max_deg)
                other.c_copy(g)

            else:

                g.set_max_deg(self.get_max_deg())
                self.c_copy(g)

            g_cont = g.content()

            if g.c_get_coef(g._deg) < 0:
                g_cont = -g_cont

            g.set_lcd(1)
            g.c_divide_int(g_cont)

            return 0

        if self._deg > other._deg:
            
            b = self
            a = other

        else:

            b = other
            a = self

        r = IntPolynomial(0)
        q = IntPolynomial(b._deg - a._deg)
        b.c_divide(a, q, r)

        while r._deg != -1:

            r.c_divide_int(r.content())
            r.set_lcd(1)
            b = a
            a = r
            r = IntPolynomial(0)
            q = IntPolynomial(b._deg - a._deg)
            b.c_divide(a, q, r)

        g.set_max_deg(a._deg)
        a.c_copy(g)
        g._lcd = 1
        g_cont = g.content()

        if g._ro_coefs[g._deg] < 0:
            sign = -1

        else:
            sign = 1

        g.c_divide_int(sign * g_cont)

        return 0

    cdef ERR_t c_sqfree_fact(self, IntPolynomialArray fact) except -1:

        cdef IntPolynomial f, fp, a, b, new_b, bp, c, d, r, cont_poly
        cdef INDEX_t i
        cdef DEG_t b_deg_diff
        cdef COEF_t cont

        self._check_is_set_raise()
        fact._check_is_not_set_raise()

        if self._deg == -1:
            raise ValueError("cannot calculate square-free factorization of zero poly.")

        if fact._max_deg != self._max_deg:
            raise ValueError("`fact` has wrong max degree.")

        fact.empty(self._max_deg + 1)
        f = IntPolynomial(self._max_deg)
        self.c_copy(f)
        cont = f.content()

        if f.c_get_coef(f._deg) < 0:
            cont = -cont

        f.c_divide_int(cont)
        cont_poly = IntPolynomial(self._max_deg)
        cont_poly.zero_poly()
        cont_poly.c_set_coef(0, cont)
        fact.append(cont_poly)

        if self._deg == 0:
            return 0

        fp = IntPolynomial(f._deg - 1)
        f.c_deriv(fp)
        a = IntPolynomial(0)
        f.c_gcd(fp, a)
        b = IntPolynomial(f._deg - a._deg)
        r = IntPolynomial(0)
        f.c_divide(a, b, r)
        b_deg_diff = f._deg - b._deg
        c = IntPolynomial(fp._deg - a._deg)
        r = IntPolynomial(0)
        fp.c_divide(a, c, r)
        bp = IntPolynomial(b._deg - 1)
        b.c_deriv(bp)
        d = IntPolynomial(c._max_deg)
        c.c_subtract(bp, d)

        for i in range(1, self._max_deg + 1):

            a = IntPolynomial(0)
            b.c_gcd(d, a)
            fact.append(a)
            new_b = IntPolynomial(b._deg - a._deg)
            r = IntPolynomial(0)
            b.c_divide(a, new_b, r)
            b_deg_diff = b._deg - new_b._deg
            b = new_b

            if b._deg == 0:
                return 0

            bp = IntPolynomial(b._deg - 1)
            b.c_deriv(bp)
            c = IntPolynomial(d._deg - a._deg)
            r = IntPolynomial(0)
            d.c_divide(a, c, r)
            d = IntPolynomial(c._deg)
            c.c_subtract(bp, d)

        return 0

    cdef ERR_t c_is_sqfree(self) except -1:

        cdef IntPolynomialArray fact

        self._check_is_set_raise()

        fact = IntPolynomialArray(self._max_deg)
        self.c_sqfree_fact(fact)

        if len(fact) == 2:
            return TRUE

        else:
            return FALSE

    ###############################
    #          PY SUGAR           #

    def __eq__(self, other):

        if self.c_eq(other) == TRUE:
            return True

        else:
            return False

    def __hash__(self):

        self._check_is_set_raise()

        if self._hashed == FALSE:

            xxh = xxhash.xxh64()
            xxh.update(self._ro_coefs[:self._deg + 1])
            self._hash = xxh.digest()
            self._hashed = TRUE

        return self._hash

    def gcd(self, IntPolynomial other):

        cdef IntPolynomial g

        g = IntPolynomial(0)
        self.c_gcd(other, g)

        return g

    def sqfree_fact(self):

        cdef IntPolynomialArray fact

        fact = IntPolynomialArray(self._deg)
        self.c_sqfree_fact(fact)

        return fact

    def is_sqfree(self):

        if self.c_is_sqfree() == TRUE:
            return True

        else:
            return False

    def get_ndarray(self):

        self._check_is_set_raise()

        if self._readonly == TRUE:
            return np.asarray(self._ro_coefs)

        else:
            return np.asarray(self._rw_coefs)

    def is_irreducible(self):

        if PolynomialRing is None:
            raise NotImplementedError("must have SageMath installed.")

        else:
            return PolynomialRing(ZZ, "x")(list(np.asarray(self._ro_coefs))).is_irreducible()

    def __copy__(self):

        cdef IntPolynomial copy = IntPolynomial(self._max_deg)
        self.c_copy(copy)
        return copy

    def __str__(self):

        s = super().__str__()

        if self._lcd == 1:
            return s

        else:
            return s.replace("]", f"]/{self._lcd}")

    def deg(self):

        self._check_is_set_raise()

        return self._deg

    def __call__(self, x, calc_deriv = False):

        self.c_eval(x, TRUE if calc_deriv else FALSE)

        if calc_deriv:
            return self.last_eval, self.last_deriv

        else:
            return self.last_eval

    def deriv(self):

        cdef IntPolynomial deriv = IntPolynomial(self._deg - 1)
        self.c_deriv(deriv)
        return deriv

    def divide(self, IntPolynomial denom):

        cdef IntPolynomial quo, rem

        denom._check_is_set_raise()
        self._check_is_set_raise()

        if denom._deg > self._deg:
            quo = IntPolynomial(-1)

        else:
            quo = IntPolynomial(self._deg - denom._deg)

        rem = IntPolynomial(0)
        self.c_divide(denom, quo, rem)

        return quo, rem

    def gcd(self, IntPolynomial other):

        cdef IntPolynomial g

        g = IntPolynomial(0)
        self.c_gcd(other, g)

        return g

    cpdef DPS_t extradps(self, MPF_t x) except -1:

        if self._deg <= 0:
            raise ValueError

        return (
            2 + # just cause
            math.ceil(math.log10(self._deg)) +
            math.ceil(math.log10(self.max_abs_coef())) +
            math.ceil((self._deg - 1) * math.log10(int(x) + 1))
        )

    def roots(self):

        if self._deg <= 0:
            raise ValueError(f"polynomial degree must be positive, not {self._deg}")

        roots_and_mults = PolynomialRing(ComplexField(mpmath.mp.prec), "z")(
            list(np.asarray(self._ro_coefs, dtype = int))
        ).roots()
        roots_and_abs_and_mults = []

        for i, (root, mult) in enumerate(roots_and_mults):

            root = mpmath.mpc(root.real(), root.imag())
            roots_and_abs_and_mults.append((root, abs(root), mult))

        return roots_and_abs_and_mults

    ###############################
    #         INVALIDATED         #

    cdef ERR_t c_set_rw_array(self, COEF_t[:,:] array) except -1:
        raise NotImplementedError("Cannot call `c_set_rw_array` on `IntPolynomial`.")

    cdef ERR_t c_set_ro_array(self, const COEF_t[:,:] array) except -1:
        raise NotImplementedError("Cannot call `c_set_ro_array` on `IntPolynomial`.")

    cdef ERR_t c_set_poly(self, INDEX_t i, IntPolynomial poly) except -1:
        raise NotImplementedError("Cannot call `c_set_poly` on `IntPolynomial`.")

    cdef ERR_t c_get_poly(self, INDEX_t i, IntPolynomial poly) except -1:
        raise NotImplementedError("Cannot call `c_get_poly` on `IntPolynomial`.")

    cdef ERR_t c_set_degs(self) except -1:
        raise NotImplementedError("Cannot call `c_set_degs` on `IntPolynomial`.")

    cpdef ERR_t append(self, IntPolynomial poly) except -1:
        raise NotImplementedError("Cannot call `append` on `IntPolynomial`.")

    def __len__(self):
        raise NotImplementedError("Cannot call `len` on `IntPolynomial`.")

cdef COEF_t mv_sum(COEF_t[:] array) except? -1:

    cdef INDEX_t i
    cdef COEF_t s = 0

    for i in range(array.shape[0]):
        s += array[i]

    return s

cdef class IntPolynomialIter:

    def __init__(self, deg, sum_abs_coefs, monic = True, reciprocal = False, irreducible = False, last_poly = None):

        if not is_int(sum_abs_coefs):
            raise TypeError("`sum_abs_coefs` must be of type `int`.")

        if not is_int(deg):
            raise TypeError("`deg` must be of type `int`.")

        if last_poly is not None and not isinstance(last_poly, IntPolynomial):
            raise TypeError("`last_poly` must be of type `IntPolynomial`.")

        if sum_abs_coefs < 1:
            raise ValueError("`sum_abs_coefs` must be at least 1.")

        if deg < 0:
            raise ValueError("`deg` must be at least 0.")

        if last_poly is not None and last_poly.deg() != deg:
            raise ValueError(f"`last_poly` degree ({last_poly.deg()}) must match `deg`.")

        if last_poly is not None and last_poly.sum_abs_coef() != sum_abs_coefs:
            raise ValueError("`last_poly` sum abs coef is not equal to passed sum_abs_coef")

        if monic and deg == 0 and sum_abs_coefs > 1:
            raise ValueError(f"No monic polynomials with `deg == 0` and `sum_abs_coefs == {sum_abs_coefs}`.")

        # print('calling 0')
        self._irreducible = TRUE if irreducible else FALSE
        self._curr_it = IntPolynomialIter.init(
            deg, sum_abs_coefs, TRUE if monic else FALSE, TRUE if reciprocal else FALSE, last_poly
        )

    def __iter__(self):
        return self

    def __next__(self):

        # print('calling c')

        while True:

            e = self._curr_it.c_next()
            # print('e', e)
            if e == 0:
                # print('__next__ ret', self._curr_it._ret)
                if self._irreducible == FALSE or self._curr_it._ret.is_irreducible():
                    return self._curr_it._ret

            elif e == -1:
                # print('raising')
                raise StopIteration

            else:
                raise RuntimeError(e)

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef IntPolynomialIter init(
        DEG_t deg, COEF_t sum_abs_coef,  BOOL_t monic, BOOL_t reciprocal, IntPolynomial last_poly
    ):

        cdef IntPolynomialIter it = IntPolynomialIter.__new__(IntPolynomialIter)
        cdef IntPolynomial recur_poly
        cdef DEG_t half_deg, j
        cdef COEF_t c
        cdef first_call = TRUE if last_poly is None else FALSE
        it._deg = deg
        it._sum_abs_coef = sum_abs_coef
        it._monic = monic
        it._reciprocal = reciprocal
        it._exhausted = FALSE
        it._pos_middle_coef = TRUE
        it._first_call = TRUE if last_poly is None else FALSE
        half_deg = (it._deg + 1) // 2 - 1
        # print(deg, sum_abs_coef, monic, reciprocal, last_poly)

        if it._reciprocal == TRUE:

            if ( # a few degenerate edge cases
                (it._deg != 0 and it._sum_abs_coef == 1) or
                (it._deg == 1 and it._monic == TRUE and it._sum_abs_coef > 2) or
                (it._deg == 0 and it._monic == TRUE and it._first_call == FALSE)
            ):

                it._exhausted = TRUE
                return it

            if it._first_call == FALSE:
                # set up polynomial of with degree half_deg that we start from
                recur_poly = IntPolynomial(half_deg)
                recur_poly.zero_poly()
                recur_poly._rw_coefs[:] = last_poly._ro_coefs[it._deg - half_deg : ]
                recur_poly._deg = half_deg

            if it._deg % 2 == 0:
                # even degree polynomials have a 'middle' coefficient
                if half_deg == -1:
                    # edge case
                    # print('calling 1')
                    it._curr_it = IntPolynomialIter.init(-1, 0, FALSE, FALSE, None)

                    if it._first_call == FALSE:
                        it._curr_it._exhausted = TRUE

                elif it._first_call == TRUE:
                    # start iterator from beginning
                    # print('calling 2')
                    it._curr_it = IntPolynomialIter.init(half_deg, 1, it._monic, FALSE, None)

                else:
                    # start iterator from recur_poly
                    # print('calling 3')
                    it._curr_it = IntPolynomialIter.init(
                        half_deg, recur_poly.sum_abs_coef(), it._monic, FALSE, recur_poly
                    )

                if it._first_call == FALSE and (it._monic == FALSE or it._deg > 0):
                    # set up sign of middle coefficient and initialize previous return value
                    it._pos_middle_coef = TRUE if last_poly._ro_coefs[it._deg // 2] <= 0 else FALSE
                    it._ret = IntPolynomial(it._deg)
                    it._ret.zero_poly()
                    it._ret._rw_coefs[:] = last_poly._ro_coefs
                    # print('it._pos_middle_coef', it._pos_middle_coef)
                        # must advance it._curr_it once if last_poly had a positive middle coefficient
                    # it._curr_it.c_next()

            else:

                if it._sum_abs_coef % 2 == 1:
                    it._exhausted = TRUE

                elif it._first_call == TRUE:
                    # print('calling 4')
                    it._curr_it = IntPolynomialIter.init(
                        half_deg, it._sum_abs_coef // 2, it._monic, FALSE, None
                    )

                else:
                    # print('calling 5')
                    it._curr_it = IntPolynomialIter.init(
                        half_deg, it._sum_abs_coef // 2, it._monic, FALSE, recur_poly
                    )


        elif it._monic == TRUE:

            if it._first_call == TRUE:

                if it._sum_abs_coef == 1:
                    # print('calling 6')
                    it._curr_it = IntPolynomialIter.init(-1, 0, FALSE, FALSE, None)

                else:
                    # print('calling 7')
                    it._curr_it = IntPolynomialIter.init(0, it._sum_abs_coef - 1, FALSE, FALSE, None)

            else:

                for j in range(it._deg - 1, -1, -1):

                    if last_poly._ro_coefs[j] != 0:
                        break # j loop

                else:
                    it._exhausted = TRUE

                recur_poly = IntPolynomial(j)
                recur_poly.zero_poly()
                recur_poly._rw_coefs[:] = last_poly._ro_coefs[ : j + 1]
                recur_poly._deg = j
                # print('calling 8')
                it._curr_it = IntPolynomialIter.init(recur_poly._deg, it._sum_abs_coef - 1, FALSE, FALSE, recur_poly)

        else:

            it._curr_it = it

            if it._deg != -1:

                if it._first_call == TRUE:

                    it._curr = np.zeros(it._deg + 1, dtype = COEF_DTYPE)
                    it._nonzero_indices = np.empty(it._deg + 1, dtype = DEG_DTYPE)

                    if it._deg == 0:

                        it._curr[0] = it._sum_abs_coef
                        it._num_nonzeros = 1
                        it._nonzero_indices[0] = 0

                    else:

                        it._curr[0] = it._sum_abs_coef - 1
                        it._curr[it._deg] = 1

                        if it._sum_abs_coef > 1:

                            it._nonzero_indices[0] = 0
                            it._nonzero_indices[1] = it._deg
                            it._num_nonzeros = 2

                        else:

                            it._nonzero_indices[0] = it._deg
                            it._num_nonzeros = 1

                    it._leftover = it._sum_abs_coef - 1
                    it._sign_index = 0
                    it._max_sign_index = (1 << it._num_nonzeros) - 1

                else:

                    it._curr = np.abs(last_poly._ro_coefs)
                    it._nonzero_indices = np.empty(it._deg + 1, dtype = DEG_DTYPE)

                    for j in range(it._deg + 1):

                        c = last_poly._ro_coefs[j]

                        if c < 0:
                            it._sign_index += 1 << it._num_nonzeros

                        if c != 0:

                            it._nonzero_indices[it._num_nonzeros] = j
                            it._num_nonzeros += 1

                    it._max_sign_index = (1 << it._num_nonzeros) - 1
                    it._leftover = it._curr[0]
                    # print('it._curr', np.asarray(it._curr))
                    # print('it._nonzero_indices', np.asarray(it._nonzero_indices))
                    # print('it._num_nonzeros', it._num_nonzeros)
                    # print('it._sign_index', it._sign_index)
                    # print('it._max_sign_index', it._max_sign_index)

        return it

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_next(self) except *:

        cdef ERR_t e
        cdef IntPolynomial last_ret_copy
        cdef COEF_t middle_coef

        if self._exhausted == TRUE:
            # print('hi1')
            return -1

        elif self._pos_middle_coef == FALSE:

            # print('hi2')
            last_ret_copy = IntPolynomial(self._deg)
            last_ret_copy.zero_poly()
            last_ret_copy._rw_coefs[:] = self._ret._ro_coefs
            self._ret = IntPolynomial(self._deg)
            self._ret.zero_poly()
            self._ret._deg = self._deg
            self._ret._rw_coefs[:] = last_ret_copy._ro_coefs
            self._ret._rw_coefs[self._deg // 2] = -self._ret._ro_coefs[self._deg // 2]
            self._pos_middle_coef = TRUE
            return 0

        elif self._curr_it is self:
            # print('calling a', self._deg, self._sum_abs_coef)
            e = self.c_next_helper()
            # print('a e', e, self._deg, self._sum_abs_coef)

            if e == -1:
                self._exhausted = TRUE

            return e

        else:

            # print('calling b', self._deg, self._sum_abs_coef)
            e = self._curr_it.c_next()
            # print('b e', e, self._deg, self._sum_abs_coef)

            if e == 0:
                # dress self._curr_it._ret
                self._ret = IntPolynomial(self._deg)
                self._ret.zero_poly()
                self._ret._deg = self._deg

                if self._reciprocal == TRUE:

                    if self._curr_it._deg >= 0:

                        self._ret._rw_coefs[ : self._curr_it._deg + 1] = self._curr_it._ret._ro_coefs[::-1]
                        self._ret._rw_coefs[self._deg - self._curr_it._deg:] = self._curr_it._ret._ro_coefs[:]

                    if self._deg % 2 == 0:

                        middle_coef = self._sum_abs_coef - 2 * self._curr_it._sum_abs_coef
                        # print('self._sum_abs_coef', self._sum_abs_coef)
                        # print('self._curr_it._sum_abs_coef',self._curr_it._sum_abs_coef)
                        # print('self._pos_middle_coef', self._pos_middle_coef)
                        self._ret._rw_coefs[self._deg // 2] = middle_coef
                        # print('self._ret._rw_coefs[self._deg // 2]', self._ret._rw_coefs[self._deg // 2])
                        if (self._monic == FALSE or self._deg > 0) and middle_coef > 0:
                            self._pos_middle_coef = FALSE

                    # print('self._ret', self._ret)

                elif self._monic == TRUE:

                    self._ret._rw_coefs[self._deg] = 1
                    # print('self._deg', self._deg)
                    # print('self._ret', self._ret)
                    # print('self._curr_it._deg', self._curr_it._deg)
                    # print('self._curr_it._ret', self._curr_it._ret)
                    # print('np.asarray(self._curr_it._ret._ro_coefs)', np.asarray(self._curr_it._ret._ro_coefs))

                    if self._curr_it._deg >= 0:

                        try:
                            self._ret._rw_coefs[ : self._curr_it._deg + 1] = self._curr_it._ret._ro_coefs

                        except ValueError:
                            # print(self._deg)
                            # print(self._sum_abs_coef)
                            # print(self._curr_it._deg)
                            # print(self._curr_it._sum_abs_coef)
                            raise

                    # print('self._ret', self._ret)

                else:
                    return 1

                return 0

            else:
                # increment boundary conditions
                if self._reciprocal == TRUE:

                    if self._deg % 2 == 0:

                        # print('self._curr_it._deg', self._curr_it._deg)
                        # print('self._curr_it._sum_abs_coef', self._curr_it._sum_abs_coef)
                        # print('self._curr_it._monic',  self._curr_it._monic)
                        # print('self._curr_it._reciprocal', self._curr_it._reciprocal)
                        # print('self._deg', self._deg)
                        # print('self._sum_abs_coef', self._sum_abs_coef)
                        # print('self._monic', self._monic)
                        # print('self._reciprocal', self._reciprocal)

                        if (
                            self._curr_it._deg == -1 or
                            (self._monic and self._curr_it._deg == 0) or
                            (2 * (1 + self._curr_it._sum_abs_coef) > self._sum_abs_coef)
                        ):
                            return -1

                        else:

                            # print('calling 9')
                            self._curr_it = IntPolynomialIter.init(
                                self._curr_it._deg, self._curr_it._sum_abs_coef + 1, self._monic, FALSE, None
                            )
                            # print('calling d')
                            e = self.c_next()
                            # print('d e', e, self._deg, self._sum_abs_coef)

                    else:

                        self._exhausted = TRUE
                        return -1

                elif self._monic == TRUE:

                    if self._curr_it._deg >= self._deg - 1 or self._curr_it._deg == -1:
                        # deg maximal
                        self._exhausted = TRUE
                        return -1

                    else:
                        # deg can be incremented
                        # print('calling 10')
                        self._curr_it = IntPolynomialIter.init(
                            self._curr_it._deg + 1, self._sum_abs_coef - 1, FALSE, FALSE, None
                        )
                        # print('calling d')
                        return self.c_next()

                else:
                    return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ERR_t c_next_helper(self) except *:

        cdef DEG_t j, n
        self._ret = IntPolynomial(self._deg)
        self._ret.zero_poly()
        self._ret._deg = self._deg

        if self._first_call == TRUE:

            if self._deg != -1:
                self._ret._rw_coefs[:] = self._curr

            self._first_call = FALSE
            return 0

        elif self._deg != -1:

            if self._sign_index >= self._max_sign_index:

                # print('I shouldn\'t be here!')

                if self._deg == 0:
                    return -1

                self._leftover -= 1
                self._curr[1] += 1

                for j in range(1, self._deg + 1):

                    if self._leftover >= 0:
                        break # j loop

                    elif j < self._deg:

                        self._leftover += self._curr[j] - 1
                        self._curr[j] = 0
                        self._curr[j + 1] += 1

                    else:
                        return -1

                self._curr[0] = self._leftover
                self._num_nonzeros = 0

                for j in range(self._deg + 1):

                    if self._curr[j] != 0:

                        self._nonzero_indices[self._num_nonzeros] = j
                        self._num_nonzeros += 1

                self._max_sign_index = (1 << self._num_nonzeros) - 1
                self._sign_index = 0

            else:
                # print('I should be here!')
                self._sign_index += 1

            self._ret._rw_coefs[:] = self._curr

            for j in range(self._num_nonzeros):

                if (self._sign_index >> j) & 1 == 1:

                    n = self._nonzero_indices[j]
                    self._ret._rw_coefs[n] = - self._curr[n]

            return 0

        else:
            return -1

