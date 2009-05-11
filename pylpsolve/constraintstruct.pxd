# This file defines a constraint struct that allows for buffered
# adding of constraints.  This shields the user from many of the
# lower-level considerations with lp_solve

from numpy cimport ndarray as ar
from numpy cimport int32_t, uint32_t, int64_t, uint64_t, float_t
from numpy import int32,uint32,int64,uint64

from types cimport isnumeric

cdef extern from "Python.h":
    void* malloc "PyMem_Malloc"(size_t n)
    void* realloc "PyMem_Realloc"(void *p, size_t n)
    void free "PyMem_Free"(void *p)

cdef extern from "lpsolve/lp_lib.h":
    ctypedef void lprec

    ecode set_rowex(lprec *lp, int row_no, int count, real *row, int *colno)
    ecode set_row(lprec *lp, int row_no, real *row)
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
    # Dealing with the indices; if index_range_mode is true, then the
    # indices refer to a range rather than individual indices
    cdef int *indices
    cdef int index_range_start  # the first index of the length n
                                # block that is the indices. If
                                # negative; not used.

    # The values
    cdef double *values

    # The total size
    cdef size_t n
    cdef int ctype
    cdef double rhs1, rhs2
    cdef size_t row_idx

########################################
# Methods to deal with this constraint

cdef inline setupConstraint(_Constraint* cstr, size_t row_idx, ar idx, ar row, str ctypestr, rhs):

    # see if we need to clear things
    if cstr.n != 0:
        clearConstraint(cstr)

    ########################################
    # Check possible bad configurations of ctype, rhs
    cstr.ctype = getCType(ctypestr)

    if cstr.ctype in [constraint_leq, constraint_geq, constraint_eq]:
        if not isnumeric(rhs):
            raise TypeError("Constraint type '%s' requires right hand side to be scalar." % ctypestr)

        cstr.rhs1 = <double>rhs
    elif cstr.ctype == constraint_in:
        if type(rhs) is tuple and len(<tuple>rhs) == 2:
            cstr.rhs1, cstr.rhs2 = (<tuple>rhs)
        elif type(rhs) is list and len(<list>rhs) == 2:
            cstr.rhs1, cstr.rhs2 = (<list>rhs)
        else:
            raise TypeError("Constraint type '%s' requires right hand side to be either 2-tuple or 2-list." % ctypestr)
    else:
        assert False
    
    ########################################
    # Now that these tests pass, copy all the values in.

    cdef bint fill_mode = False
    cdef double fill_value = 0

    # Initializing stuff having to deal with the range indexing
    cdef bint idx_range_mode = False
    cdef int il, iu
    cstr.index_range_start = -1  # indicating not used

    # Determine the size
    if idx is not None:

        if idx.ndim == 1:
            if idx.shape[0] != 1 and row.shape[0] == 1:
                cstr.n = idx.shape[0]
                fill_mode = True
                fill_value = row[0]
            elif idx.shape[0] == row.shape[0]:
                cstr.n = idx.shape[0]
            else:
                assert False
        elif idx.ndim == 2:  
            # this means it defines a range instead of individual values
            assert idx.shape[0] == 1
            assert idx.shape[1] == 2

            il = idx[0,0]
            iu = idx[0,1]
            assert iu > il

            cstr.n = iu - il
            cstr.index_range_start = il + 1  # for 1-based counting of indices
            idx_range_mode = True
        else:
            assert False

    else:
        cstr.n = row.shape[0]

    # Set the indices
    if idx is None or idx_range_mode:
        cstr.indices = NULL
    else:
        cstr.indices = <int *> malloc((cstr.n+1)*sizeof(int))

        if cstr.indices == NULL:
            raise MemoryError

        copyIntoIndices(cstr, idx)

    # Set the values
    cstr.values = <double*> malloc((cstr.n+1)*sizeof(double))

    if cstr.values == NULL:
        raise MemoryError

    if fill_mode:
        fillValues(cstr, fill_value)
    else:
        copyIntoValues(cstr, row)


############################################################
# Reliably copying in the indices and values; need to handle all the
# possible types.  For values, it's npfloat.  For the others, any
# integer should work.


########################################
# Values

cdef inline copyIntoValues(_Constraint* cstr, ar a_o):

    cdef size_t i
    cdef ar[float_t] a = a_o

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.values[i+1] = a[i]

cdef inline void fillValues(_Constraint* cstr, double v, size_t n):

    cdef size_t i

    for 1 <= i <= n:
        cstr.values[i] = v

########################################
# Now handling the index buffer

cdef inline copyIntoIndices(_Constraint* cstr, ar a):

    dt = a.dtype
    if dt is int32:      copyIntoIndices_int32(cstr, a)
    elif dt is uint32:   copyIntoIndices_uint32(cstr, a)
    elif dt is int64:    copyIntoIndices_int64(cstr, a)
    elif dt is uint64:   copyIntoIndices_uint64(cstr, a)
    else:                
        if sizeof(int) == 4:
            ua = uint32(a)
            if not (ua == a).all():
                raise TypeError("Error converting index array to 32bit integers.")

            copyIntoIndices_uint32(cstr, ua)
        else:
            ua = uint64(a)
            if not (ua == a).all():
                raise TypeError("Error converting index array to 64bit integers.")

            copyIntoIndices_uint64(cstr, ua)

            
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

cdef inline setInLP(_Constraint *cstr, lprec* lp, size_t n_cols, int *countrange):

    if self.n == 0:
        return

    # Ensure that the columns and all are sized up correctly
    if cstr.indices == NULL and cstr.n != n_cols:
        _setIndicesToRange(cstr)

    # Vanila constraint
    if cstr.ctype in [constraint_leq, constraint_geq, constraint_equal]:
        _setRow(cstr, lp, cstr.row_idx, cstr.ctype, cstr.rhs1, countrange)

    # range constraint
    elif cstr.ctype == constraint_in:
        if cstr.rhs1 < cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs2, countrange)
            set_rh_range(lp,  cstr.row_idx, cstr.rhs2 - cstr.rhs1)
        elif cstr.rhs1 > cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs1, countrange)
            set_rh_range(lp,  cstr.row_idx, cstr.rhs1 - cstr.rhs2)
        else:
            _setRow(cstr, lp, constraint_eq, cstr.rhs1, countrange)
    else:
        assert False  # no silent fail


cdef inline _setRow(_Constraint *cstr, lp, int ctype, double rhs, int *countrange):

    # Need to accommidate the start-at-1 indexing
    if cstr.indices == NULL:
        if cstr.index_range_start != -1:
            set_row_ex(lp, cstr.row_idx, cstr.n, cstr.values+1, &countrange[cstr.index_range_start-1] )
        else:
            set_row(lp, cstr.row_idx, cstr.values)
    else:
        set_row_ex(lp, cstr.row_idx, cstr.values, cstr.n, cstr.values+1, cstr.indices+1)

    set_constr_type(lp, cstr.row_idx, ctype)
    set_rh(lp, cstr.row_idx, rhs)


cdef inline _setIndicesToRange(_Constraint *cstr):
    cdef size_t i

    # all we need to do now that we're using range based indexing
    cstr.index_range_start = 1


cdef inline void clearConstraint(_Constraint *cstr):
    if cstr.indices != NULL: free(cstr.indices)
    if cstr.values  != NULL: free(cstr.indices)
    
    cstr.indices = NULL
    cstr.values = NULL
    cstr.n = 0

cdef inline bint inUse(_Constraint *cstr):
    return (cstr.n != 0)
