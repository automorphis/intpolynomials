cimport numpy as cnp

ctypedef cnp.longlong_t     COEF_t
ctypedef cnp.int_t          DEG_t
ctypedef object             MPF_t
ctypedef cnp.int_t          INDEX_t
ctypedef cnp.int_t          DPS_t
ctypedef int                ERR_t
ctypedef int                BOOL_t

cdef ERR_t is_readonly(cnp.ndarray array) except -1

cdef DEG_t calc_deg(const COEF_t[:,:] array, INDEX_t i) except? -1

cdef class Int_Polynomial_Array:

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

    cdef ERR_t c_set_poly(self, INDEX_t i, Int_Polynomial poly) except -1

    cdef ERR_t c_get_poly(self, INDEX_t i, Int_Polynomial poly) except -1

    cdef ERR_t c_set_degs(self) except -1

    cpdef ERR_t empty(self, INDEX_t max_len) except -1


    cdef ERR_t c_copy(self, Int_Polynomial_Array copy) except -1

    cpdef COEF_t max_abs_coef(self) except -1

    cdef ERR_t c_eq(self, Int_Polynomial_Array other) except -1

    cpdef ERR_t append(self, Int_Polynomial poly) except -1

cdef class Int_Polynomial(Int_Polynomial_Array):

    cdef:
        INDEX_t _index
        DEG_t _deg
        const COEF_t[:] _ro_coefs
        COEF_t[:] _rw_coefs
        MPF_t last_eval
        MPF_t last_deriv

    cdef ERR_t c_set_array(self, Int_Polynomial_Array array, INDEX_t index) except -1

    cdef ERR_t c_set_coef(self, DEG_t j, COEF_t c) except -1

    cdef COEF_t c_get_coef(self, DEG_t j) except? -1

    cpdef ERR_t zero_poly(self) except -1


    cdef ERR_t c_copy(self, Int_Polynomial_Array copy) except -1

    cdef ERR_t c_eval(self, MPF_t x, BOOL_t calc_deriv) except -1
