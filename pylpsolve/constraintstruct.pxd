# This file defines a constraint struct that allows for buffered
# adding of constraints.  This shields the user from many of the
# lower-level considerations with lp_solve

from numpy cimport ndarray as ar
from numpy cimport float32_t, float64_t, int32_t, uint32_t, int64_t, uint64_t
from numpy import float32,float64,int32,uint32,int64,uint64

cdef extern from "Python.h":
    void* malloc "PyMem_Malloc"(size_t n)
    void* realloc "PyMem_Realloc"(void *p, size_t n)
    void free "PyMem_Free"(void *p)

cdef extern from "lpsolve/lp_lib.h":
    ctypedef void lprec

    ecode set_rowex(lprec *lp, int row_no, int count, real *row, int *colno)
    ecode set_constr_type(lprec *lp, int row, int con_type)
    ecode set_rh(lprec *lp, int row, real value)
    ecode set_rh_range(lprec *lp, int row, real deltavalue)

############################################################
# Constraint types

DEF constraint_free    = 0
DEF constraint_leq     = 1
DEF constraint_geq     = 2
DEF constraint_equal   = 3
DEF constraint_between = 4

cdef list _constraint_type_list = [
    ("<"      , constraint_leq),
    ("<="     , constraint_leq),
    ("=<"     , constraint_leq),
    ("leq"    , constraint_leq),
    ("lt"     , constraint_leq),
    (">"      , constraint_geq),
    (">="     , constraint_geq),
    ("=>"     , constraint_geq),
    ("geq"    , constraint_geq),
    ("gt"     , constraint_geq),
    ("="      , constraint_equal),
    ("=="     , constraint_equal),
    ("eq"     , constraint_equal),
    ("equal"  , constraint_equal),
    ("in"     , constraint_in),
    ("between", constraint_in),
    ("range"  , constraint_in)]

cdef dict _constraint_map = dict(_constraint_type_list)
cdef str _valid_constraint_identifiers = \
    ','.join(["'%s'" % cid for cid,ct in _constraint_type_list])


cdef getCType(str ctypestr):
    try:
        return _constraint_map[ctypestr]
    except KeyError:
        raise ValueError("Constraint type '%s' not recognized." % ctypestr)

######################################################################
# Now functions and a struct dealing with the constraints

cdef struct _Constraint:
    cdef int *indices
    cdef double *values
    cdef size_t n
    cdef int ctype
    cdef double rhs1, rhs2
    cdef size_t row_idx

cdef inline setupConstraint(_Constraint* cstr, size_t row_idx, ar idx, ar row, str ctypestr, rhs):

    # see if we need to clear things
    if cstr.n != 0:
        clearConstraint(cstr)

    assert row.ndim == 1

    cstr.n = row.shape[0]

    cstr.row_idx = row_idx

    ########################################
    # Check possible bad configurations of ctype, rhs
    cstr.ctype = getCType(ctypestr)

    if cstr.ctype in [constraint_leq, constraint_geq, constraint_eq]:
        assert type(rhs) is float
        cstr.rhs1 = rhs
    elif cstr.ctype == constraint_in:
        assert type(rhs) is tuple
        assert len(<tuple>rhs) == 2

        cstr.rhs1, cstr.rhs2 = (<tuple>rhs)

    # Now that these tests pass, copy all the values in.
    cstr.values = <double*> malloc((cstr.n+1)*sizeof(double))

    if cstr.values == NULL:
        raise MemoryError

    cstr.values[0] = 0

    if idx is None:
        cstr.indices = NULL
        cstr.copyIntoValues(row, True)
    else:
        cstr.indices = <int *> malloc((cstr.n+1)*sizeof(int))

        if cstr.indices == NULL:
            raise MemoryError

        cstr.copyIntoIndices(idx)
        cstr.copyIntoValues(row, False)


############################################################
# Reliably copying in the indices and values; need to handle all
# the possible types

 ########################################
# Values

cdef inline copyIntoValues(_Constraint* cstr, ar a):

    dt = a.dtype

    if dt is float32:
        copyIntoValues_float32(cstr, a)
    elif dt is float64:
        copyIntoValues_float64(cstr, a)
    else:
        copyIntoValues_float64(cstr, float64(a))

cdef inline copyIntoValues_float32(_Constraint* cstr, ar a_float32_o):
    cdef size_t i
    cdef ar[float32_t] a_float32 = a_float32_o

    with cython.boundscheck(False):
        for 0 <= i < a_float32.shape[0]:
            cstr.values[i+1] = a_float32[i]


cdef inline copyIntoValues_float64(_Constraint* cstr, ar a_float64_o):
    cdef size_t i
    cdef ar[float64_t] a_float64 = a_float64_o

    with cython.boundscheck(False):
        for 0 <= i < a_float64.shape[0]:
            cstr.values[i+1] = a_float64[i]


########################################
# Now handling the index buffer

cdef inline copyIntoIndices(_Constraint* cstr, ar a):

    dt = a.dtype
    if dt is int32:      copyIntoIndices_int32(cstr, a)
    elif dt is uint32:   copyIntoIndices_uint32(cstr, a)
    elif dt is int64:    copyIntoIndices_int64(cstr, a)
    elif dt is uint64:   copyIntoIndices_uint64(cstr, a)
    else:                copyIntoIndices_uint32(cstr, uint32(a))

cdef inline copyIntoIndices_int32(_Constraint *cstr, ar a_o):

    cdef ar[int32_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.intbuf[i+1] = a[i] + 1

cdef inline copyIntoIndices_uint32(_Constraint *cstr, ar a_o):

    cdef ar[uint32_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.intbuf[i+1] = a[i] + 1

cdef inline copyIntoIndices_int64(_Constraint *cstr, ar a_o):

    cdef ar[int64_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.intbuf[i+1] = a[i] + 1

cdef inline copyIntoIndices_uint64(_Constraint *cstr, ar a_o):

    cdef ar[uint64_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.intbuf[i+1] = a[i] + 1

############################################################
# Now the routines to add this to the model

cdef inline setInLP(_Constraint *cstr, lprec* lp, size_t n_cols):

    if self.n == 0:
        return

    # Ensure that the columns and all are sized up correctly
    if cstr.indices == NULL and cstr.n != n_cols:
        _setIndicesToRange(cstr)

    # Next add the constraint

    # Vanila constraint
    if cstr.ctype in [constraint_leq, constraint_geq, constraint_equal]:
        _setRow(cstr, lp, cstr.row_idx, cstr.ctype, cstr.rhs1)

    # range constraint
    elif cstr.ctype == constraint_in:
        if cstr.rhs1 < cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs2)
            set_rh_range(lp,  cstr.row_idx, cstr.rhs2 - cstr.rhs1)
        elif cstr.rhs1 > cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs1)
            set_rh_range(lp,  cstr.row_idx, cstr.rhs1 - cstr.rhs2)
        else:
            _setRow(cstr, lp, constraint_eq, cstr.rhs1)
    else:
        assert False  # no silent fail

cdef inline _setRow(_Constraint *cstr, lp, int ctype, double rhs):
    set_row_ex(lp, cstr.row_idx, cstr.n, cstr.values, cstr.indices)
    set_constr_type(lp, cstr.row_idx, ctype)
    set_rh(lp, cstr.row_idx, rhs)

cdef inline _setIndicesToRange(_Constraint *cstr):
    cdef size_t i

    # Need to reset the indices to be the first n indices
    cstr.indices = <int*>malloc( (cstr.n + 1)*sizeof(int))

    if cstr.indices == NULL:
        raise MemoryError

    for 1 <= i <= cstr.n:
        cstr.indices[i] = i

cdef inline void clearConstraint(_Constraint *cstr):
    if cstr.indices != NULL: free(cstr.indices)
    if cstr.values  != NULL: free(cstr.indices)
    
    cstr.indices = NULL
    cstr.values = NULL
    cstr.n = 0

cdef inline bint inUse(_Constraint *cstr):
    return (cstr.n != 0)
