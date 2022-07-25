cimport numpy as cnp

ctypedef cnp.longlong_t     COEF_t
ctypedef cnp.int_t          DEG_t
ctypedef object             MPF_t
ctypedef cnp.int_t          INDEX_t
ctypedef cnp.int_t          DPS_t

cdef enum BOOL_TYPE:
    TRUE, FALSE

cdef class Int_Polynomial:

    cdef:
        COEF_t[:] _coefs
        DPS_t _dps
        DEG_t _deg
        DEG_t _max_deg
        BOOL_TYPE _is_natural
        MPF_t last_eval
        MPF_t last_eval_deriv
        DEG_t _start_index

    cdef void _init(self, coefs, DPS_t dps, BOOL_TYPE is_natural) except *

    cpdef DEG_t get_deg(self)

    cdef COEF_t[:] get_coefs_mv(self)

    cpdef DPS_t get_dps(self)

    cpdef DEG_t get_max_deg(self)

    cpdef Int_Polynomial trim(self)

    cpdef cnp.ndarray[COEF_t, ndim=1] ndarray_coefs(self, natural_order = ?, include_hidden_coefs = ?)

    cpdef COEF_t max_abs_coef(self)

    cdef void _c_eval_both(self, MPF_t x, BOOL_TYPE calc_deriv)

    cdef void c_eval_both(self, MPF_t x)

    cdef void c_eval_only(self, MPF_t x)

    cdef BOOL_TYPE eq(self, Int_Polynomial other)

    cdef void set_coef(self, DEG_t i, COEF_t coef) except *

    cdef COEF_t get_coef(self, DEG_t i) except? -1

cdef class Int_Polynomial_Array:

    cdef:
        INDEX_t _max_size
        COEF_t[:,:] _array
        INDEX_t _curr_index
        DPS_t _dps
        DEG_t _max_deg

    cpdef void init_empty(self, INDEX_t init_size) except *

    cdef void init_from_mv(self, COEF_t[:,:] mv, INDEX_t size)

    cdef INDEX_t get_len(self)

    cdef BOOL_TYPE eq(self, Int_Polynomial_Array other)

    cdef void set_curr_index(self, INDEX_t curr_index)

    cdef INDEX_t get_curr_index(self)

    cpdef DEG_t get_max_deg(self)

    cpdef void append(self, Int_Polynomial poly) except *

    cpdef void pad(self, INDEX_t pad_size) except *

    cpdef Int_Polynomial get_poly(self, INDEX_t i)

    cpdef cnp.ndarray[COEF_t, ndim = 2] get_ndarray(self)