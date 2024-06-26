cimport numpy as cnp

ctypedef object             MPF_t
ctypedef int                ERR_t
ctypedef int                BOOL_t
ctypedef cnp.longlong_t     COEF_t

IF UNAME_SYSNAME == "Windows":

    ctypedef cnp.int_t      DEG_t
    ctypedef cnp.int_t      INDEX_t
    ctypedef cnp.int_t      DPS_t

ELIF UNAME_SYSNAME == "Linux":

    ctypedef cnp.longlong_t     DEG_t
    ctypedef cnp.longlong_t     INDEX_t
    ctypedef cnp.longlong_t     DPS_t


cdef COEF_t gcd(COEF_t a, COEF_t b) except -1

cdef COEF_t mv_gcd(const COEF_t[:] coefs) except -1

cdef ERR_t is_readonly(cnp.ndarray array) except -1

cdef DEG_t calc_deg(const COEF_t[:,:] array, INDEX_t i) except? -1

cdef class IntPolynomialArray:

    cdef:
        BOOL_t _is_set
        BOOL_t _degs_set
        BOOL_t _readonly
        INDEX_t _curr_index
        INDEX_t _max_len
        DEG_t _max_deg
        DEG_t[:] _degs
        const COEF_t[:,:] _ro_array
        COEF_t[:,:] _rw_array

    cdef inline ERR_t _check_is_set_raise(self) except -1

    cdef inline ERR_t _check_is_not_set_raise(self) except -1

    cdef inline ERR_t _check_readwrite_raise(self) except -1

    cdef inline ERR_t _check_i_raise(self, INDEX_t i) except -1

    cdef inline ERR_t _check_j_raise(self, DEG_t j) except -1

    cdef inline ERR_t _check_degs_set_raise(self) except -1

    cdef ERR_t c_set_rw_array(self, COEF_t[:,:] mv) except -1

    cdef ERR_t c_set_ro_array(self, const COEF_t[:,:] mv) except -1

    cdef ERR_t c_set_poly(self, INDEX_t i, IntPolynomial poly) except -1

    cdef ERR_t c_get_poly(self, INDEX_t i, IntPolynomial poly) except -1

    cdef ERR_t c_set_degs(self) except -1

    cpdef ERR_t empty(self, INDEX_t max_len) except -1

    cpdef ERR_t zeros(self, INDEX_t max_len) except -1

    cpdef ERR_t ones(self, INDEX_t max_len) except -1

    cpdef ERR_t clear(self) except -1

    cpdef ERR_t set_max_deg(self, DEG_t max_deg) except -1

    cpdef ERR_t get_max_deg(self) except? -1

    cdef ERR_t c_copy(self, IntPolynomialArray copy) except -1

    cpdef COEF_t max_abs_coef(self) except -1

    cdef ERR_t c_eq(self, IntPolynomialArray other) except -1

    cpdef ERR_t append(self, IntPolynomial poly) except -1

    cdef INDEX_t c_len(self) except -1

cdef class IntPolynomialArray_Iter:

    cdef INDEX_t i
    cdef IntPolynomialArray array

cdef class IntPolynomial(IntPolynomialArray):

    cdef:
        INDEX_t _index
        DEG_t _deg
        const COEF_t[:] _ro_coefs
        COEF_t[:] _rw_coefs
        MPF_t last_eval
        MPF_t last_deriv
        bytes _hash
        BOOL_t _hashed
        COEF_t _lcd

    cdef ERR_t c_set_array(self, IntPolynomialArray array, INDEX_t index) except -1

    cdef ERR_t c_set_coef(self, DEG_t j, COEF_t c) except -1

    cdef COEF_t c_get_coef(self, DEG_t j) except? -1

    cpdef ERR_t zero_poly(self) except -1

    cpdef ERR_t set_lcd(self, COEF_t lcd) except -1

    cpdef ERR_t get_lcd(self) except -1

    cdef ERR_t c_copy(self, IntPolynomialArray copy) except -1

    cdef ERR_t c_eval(self, MPF_t x, BOOL_t calc_deriv) except -1

    cdef ERR_t c_deriv(self, IntPolynomial deriv) except -1

    cpdef COEF_t content(self) except -1

    cpdef COEF_t sum_abs_coef(self) except -1

    # cdef ERR_t c_roots(self, object roots_and_roots_abs, INDEX_t max_steps, INDEX_t max_restarts) except -1

    cdef ERR_t c_divide(self, IntPolynomial denom, IntPolynomial quo, IntPolynomial rem) except -1

    cdef ERR_t c_divide_int(self, COEF_t c) except -1

    cdef ERR_t c_subtract(self, IntPolynomial subtra, IntPolynomial diff) except -1

    cdef ERR_t c_add(self, IntPolynomial addend, IntPolynomial _sum) except -1

    cdef ERR_t c_gcd(self, IntPolynomial other, IntPolynomial g) except -1

    cdef ERR_t c_sqfree_fact(self, IntPolynomialArray fact) except -1

    cdef ERR_t c_is_sqfree(self) except -1

    cpdef DPS_t extradps(self, MPF_t x) except -1

cdef COEF_t mv_sum(COEF_t[:] array) except? -1

cdef class IntPolynomialIter:

    cdef:
        COEF_t _sum_abs_coef
        DEG_t _deg, _num_nonzeros
        BOOL_t _first_call
        COEF_t[:] _curr
        IntPolynomial _ret
        COEF_t _leftover
        INDEX_t _sign_index, _max_sign_index
        DEG_t[:] _nonzero_indices
        BOOL_t _monic, _reciprocal
        IntPolynomialIter _curr_it
        BOOL_t _pos_middle_coef, _exhausted
        BOOL_t _irreducible


    @staticmethod
    cdef IntPolynomialIter init(
        DEG_t deg, COEF_t sum_abs_coef, BOOL_t monic, BOOL_t reciprocal, IntPolynomial last_poly
    )

    cdef ERR_t c_next(self) except *

    cdef ERR_t c_next_helper(self) except *