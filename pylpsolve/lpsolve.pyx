import numpy as np
from numpy cimport ndarray as ar
from numpy cimport float32_t, float64_t, int32_t, uint32_t, int64_t, uint64_t
from numpy cimport empty, ones, zeros

import warnings

cimport cython

ctypedef double real
ctypedef unsigned char ecode

######################################################################
# LPSolve constants

DEF constraint_leq   = 1
DEF constraint_geq   = 2
DEF constraint_equal = 3

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
    ("equal"  , constraint_equal)]

cdef dict _constraint_map = dict(_constraint_type_list)
cdef str _valid_constraint_identifiers = \
    ','.join(["'%s'" % cid for cid,ct in _constraint_type_list])


cdef extern from "lpsolve55/lp_lib.h":
    ctypedef void lprec

    cdef:
        lprec* make_lp(int rows, int cols)
        void delete_lp(lprec*)

        ecode resize_lp(lprec *lp, int rows, int columns)

        ecode set_obj_fn(lprec*, real* row)
        ecode set_obj_fnex(lprec*, int count, real*row, int *colno)

        ecode add_constraint(lprec*, real* row, int ctype, real rh)
        ecode add_constraintex(lprec*, int count, real* row, int *colno, int ctype, real rh)
        
        ecode set_add_rowmode(lprec*, unsigned char turn_on)
        
        ecode set_lowbo(lprec*, int column, real value)
        ecode set_upbo(lprec*, int column, real value)
        ecode set_bounds(lprec*, int column, real lower, real upper)
        ecode set_unbounded(lprec *lp, int column)

        void set_presolve(lprec*, int do_presolve, int maxloops)
        int get_Ncolumns(lprec*)

        ecode get_variables(lprec*, real *var)
        real get_objective(lprec*)

        int solve(lprec *lp)
        
        int print_lp(lprec *lp)


cdef extern from "stdlib.h":
    void* malloc(size_t)
    void* realloc(void*, size_t)
    void free(void*)

class LPSolveException(Exception): pass
        
# Set the presolve flags
cdef dict _presolve_flags = {
"presolve_none" :          0,
"presolve_rows" :          1,
"presolve_cols" :          2,
"presolve_lindep" :        4,
"presolve_aggregate" :     8,
"presolve_sparser" :      16,
"presolve_sos" :          32,
"presolve_reducemip" :    64,
"presolve_knapsack" :    128,
"presolve_elimeq2" :     256,
"presolve_impliedfree" : 512,
"presolve_reducegcd"   : 1024,
"presolve_probefix"    : 2048,
"presolve_probereduce" : 4096,
"presolve_rowdominate" : 8192,
"presolve_coldominate" : 16384,
"presolve_mergerows"   : 32768,
"presolve_impliedslk"  : 65536,
"presolve_colfixdual"  : 131072,
"presolve_bounds"      : 262144,
"presolve_duals"       : 524288,
"presolve_sensduals"   : 1048576}
   
cdef class LPSolve:
    """
    The class wrapping the lp_solve api 
    """

    cdef real* rbuf, *rbuf_allocated
    cdef int* intbuf, *intbuf_allocated
    cdef size_t rbuf_size, intbuf_size
    cdef int n_constraints

    cdef lprec *lp

    def __cinit__(self):
        self.lp = NULL
        self.rbuf = self.rbuf_allocated = NULL
        self.intbuf = self.intbuf_allocated = NULL

    def __dealloc__(self):
        if self.rbuf_allocated != NULL: free(self.rbuf_allocated)
        if self.intbuf_allocated != NULL: free(self.intbuf_allocated)
        if self.lp != NULL: delete_lp(self.lp)

    def __init__(self, size_t allocated_rows, size_t allocated_cols):
        self.lp = make_lp(0, allocated_cols)

        # Doing resize second means we just allocate space for that many rows, not create them.
        self.resize(allocated_rows, allocated_cols)

        if self.lp == NULL:
            raise LPSolveException("Error creating model with %d rows and %d columns."
                                   % (allocated_rows, allocated_cols))

        # Set the default options
        for k, v in _presolve_flags.iteritems():
            self.options[k] = False

    cpdef resize(self, size_t allocated_rows = 0, size_t allocated_cols = 0):
        """
        Resixes the lp.  This is analagous to lp_solve's resize function.
        """

        if allocated_rows == 0: allocated_rows = self.nRows()
        if allocated_cols == 0: allocated_cols = self.nColumns()

        if resize_lp(self.lp, allocated_rows, allocated_cols) != 1:
            raise LPSolveException("Error sizing/resizing model to %d rows and %d columns."
                                   % (allocated_rows, allocated_cols))
        

    cpdef setObjective(self, coefficients):
        """
        Sets the objective function.
        
        `coefficients` may be either a single array, a dictionary, or
        a 2-tuple with the form (index array, value array).  In the
        case of a single array, it must be 1 dimensional and have the
        same length as the number of columns in the lp.  If it is a
        dictionary, the keys of the dictionary are the indices of the
        non-zero values which are given by the corresponding values.
        If it is a (index array, value array) pair, the corresponding
        pairs have the same behavior.
        """

        cdef ar a
        cdef ar idx
        cdef dict d
        cdef size_t n, i, k

        if type(coefficients) is ar:
            a = coefficients

            if a.ndim != 1:
                if a.size == a.shape[0]:
                    a = a.ravel()
                else:
                    raise LPSolveException("Only 1-d arrays are allowed for coefficients of objective function.")

            self._check_full_sizing(a)
            self._copy_into_real_buffer(a)

            if set_obj_fn(self.lp, self.rbuf - 1) != 1:
                raise LPSolveException("Error adding objective function.")

        elif type(coefficients) is dict:
            d = coefficients
            n = len(d)

            self._resize_real_buffer(n)
            self._resize_int_buffer(n)
            
            i = 0
            for k, v in d.iteritems():
                self.intbuf[i] = k + 1
                self.rbuf[i] = v
                i += 1

            if set_obj_fnex(self.lp, n, self.rbuf, self.intbuf) != 1:
                raise LPSolveException("Error adding objective function.")

        elif (type(coefficients) is tuple or type(coefficients) is list) and len(coefficients) == 2:
            
            idx, a = coefficients
            n = idx.shape[0]

            if idx.shape[0] != a.shape[0]:
                raise LPSolveException("Index array and coefficient array do not have the same shape.")

            self._copy_into_real_buffer(a)
            self._copy_into_int_buffer_shift(idx)

            if set_obj_fnex(self.lp, n, self.rbuf, self.intbuf) != 1:
                raise LPSolveException("Error adding objective function.")
        else:
            raise LPSolveException("coefficients argument must either be a 1d array, 2-tuple, or dict.")



    cpdef addConstraint(self, coefficients, str constraint_type, rhs):
        """        
        Adds a constraint, or set of constraints to the lp.

        `coefficients` may be either a single array, a dictionary, or
        a 2-tuple with the form (index array, value array).  In the
        case of a single array, it must be either 1 or 2 dimensional
        and have the same length/number of columns as the number of
        columns in the lp.  If it is 2 dimensions, each row
        corresponds to a constraint, and `rhs` must be a vector with a
        coefficient for each row.  In this case, constraint_type must
        be the same for all rows.

        If it is a dictionary, the keys of the dictionary are the
        indices of the non-zero values which are given by the
        corresponding values.  

        If it is a (index array, value array) pair, the corresponding
        pairs have the same behavior as in the dictionary case.

        `constraint_type` is a string determining the type of
        inequality or equality in the constraint.  The following are
        all valid identifiers: '<', '<=', '=<', 'lt', 'leq'; '>',
        '>=', '=>', 'gt', 'geq', '=', '==', 'equal', and 'eq'. 
        """
        
        cdef dict d
        cdef ar row, b
        cdef ar[double] bcast
        cdef ar col_idx
        cdef size_t i, k, n, m
        cdef real v, rhs_real

        cdef int constraint_id

        try:
            constraint_id = _constraint_map[constraint_id]
        except KeyError:
            try:
                constraint_id = _constraint_map[constraint_id.lower()]
            except KeyError:
                raise TypeError("Constraint type '%s' not recognized, valid identifiers are %s."
                                % _valid_constraint_identifiers)

        if isinstance(coefficients, ar):
            row = coefficients

            # If it's the basic deal
            if row.ndim == 1:
                self._addConstraintDirect(row, constraint_id, rhs)
            elif row.size == row.shape[0]:
                self._addConstraintDirect(row.ravel(), constraint_id, rhs)
            elif row.ndim == 2:
                # The case when we're adding a whole block of constraints at once.
                m = row.shape[0]
                bcast = self._ensure_rhs_is_1d_double(rhs, m)

                for i from 0 <= i < m:
                    self._addConstraintDirect(row[i, :], constraint_id, bcast[i])
            else:
                raise LPSolveException("Coefficient matrix must be either 1d or 2d.")

        elif type(coefficients) is dict:

            # Now adding a dictionary of terms
            d = coefficients
            n = len(d)

            self._resize_real_buffer(n)
            self._resize_int_buffer(n)

            i = 0
            for k, v in d.iteritems():
                self.intbuf[i] = k + 1
                self.rbuf[i] = v
                i += 1

            self._addSparseConstraintFromBuffer(n, constraint_id, rhs)

        elif (type(coefficients) is tuple or type(coefficients) is list) and len(coefficients) == 2:
            
            col_idx, row = coefficients
    
            # Two cases for row...
            if row.ndim == 1:
                n = col_idx.shape[0]
                
                if row.shape[0] != n:
                    raise LPSolveException("Length of index array (%d) must match number of coefficients (%d)."
                                            % (col_idx.shape[0], row.shape[0]))

                self._copy_into_int_buffer_shift(col_idx)
                self._copy_into_real_buffer(row)
                self._addSparseConstraintFromBuffer(n, constraint_id, rhs)

            elif row.ndim == 2:

                if col_idx.ndim == 1:
                    n = col_idx.shape[0]

                    if row.shape[1] != col_idx.shape[0]:
                        raise LPSolveException("Length of index array (%d) must match number of coefficients (%d)."
                                                % (col_idx.shape[0], row.shape[1]))

                    self._copy_into_int_buffer_shift(col_idx)
                    m = row.shape[0]
                    bcast = self._ensure_rhs_is_1d_double(rhs, m)

                    for i from 0 <= i < m:
                        self._copy_into_real_buffer(row[i,:])
                        self._addSparseConstraintFromBuffer(n, constraint_id, bcast[i])

                elif col_idx.ndim == 2:
                    n = col_idx.shape[1]

                    if row.shape[1] != col_idx.shape[1]:
                        raise LPSolveException("Length of index array (%d) must match number of coefficients (%d)."
                                                % (col_idx.shape[1], row.shape[1]))

                    
                    m = row.shape[0]
                    bcast = self._ensure_rhs_is_1d_double(rhs, m)

                    for i from 0 <= i < m:
                        self._copy_into_int_buffer_shift(col_idx[i,:])
                        self._copy_into_real_buffer(row[i,:])
                        self._addSparseConstraintFromBuffer(n, constraint_id, bcast[i])
                else:
                    raise LPSolveException("Column index array must be either 1d or 2d.")
                    
            else:
                raise LPSolveException("Coefficient matrix in sparse must be either 1d or 2d.")
        else:
            raise LPSolveException("coefficients argument must either be a 1d array, 2-tuple, or dict.")

        
    cdef _addConstraintDirect(self, ar row, int constraint_id, real rhs):
    
        self._check_full_sizing(row)
        self._copy_into_real_buffer(row)

        cdef double t = self._normalizeRealBuffer(row.shape[0])
        
        if add_constraint(self.lp, self.rbuf-1, constraint_id, rhs/t) != 1:
            raise LPSolveException("Error adding constraint.")

    cdef _addSparseConstraintFromBuffer(self, size_t n, int constraint_id, double rhs):
        
        cdef double t = self._normalizeRealBuffer(n)
        
        if add_constraintex(self.lp, n, self.rbuf, self.intbuf, constraint_id, rhs/t) != 1:
            raise LPSolveException("Error adding constraint.")


    cdef double _normalizeRealBuffer(self, size_t n):
        cdef double t = 0
        cdef size_t i
        
        for i from 0 <= i < n:  
            t += abs(self.rbuf[i])
        
        t /= n
        assert t > 0

        for i from 0 <= i < n:  
            self.rbuf[i] /= t

        return t


    cdef ar _ensure_rhs_is_1d_double(self, rhs, size_t m):

        cdef ar b

        if type(rhs) is float or type(rhs) is int:
            return ones(m)*rhs
        elif not isinstance(rhs, ar):
            raise LPSolveException("When coefficients is a 2d array, rhs must be corresponding 1d array or a scalar.")

        b = rhs

        if b.ndim != 1:
            raise LPSolveException("Dimensions wrong on rhs array.")
        if b.shape[0] != m:
            raise LPSolveException("Size wrong on rhs array.") 
        
        return np.asarray(b, dtype=np.float)


    cpdef setUnbounded(self, size_t idx):
        """
        Sets the variable `idx` to unbounded (default is positive).
        """

        set_unbounded(self.lp, idx + 1)


    cpdef setLowerBound(self, size_t idx, real lb):
        """
        Sets the variable `idx` to unbounded (default is positive).
        """

        set_lowbo(self.lp, idx + 1, lb)


    cpdef setUpperBound(self, size_t idx, real ub):
        """
        Sets the variable `idx` to unbounded (default is positive).
        """

        set_upbo(self.lp, idx + 1, ub)


    cpdef addingConstraints(self):
        """
        Turns on row-adding mode.
        """
        
        set_add_rowmode(self.lp, True)


    cpdef doneAddingConstraints(self):
        """
        Turns off row-adding mode
        """

        set_add_rowmode(self.lp, False)
        

    def solve(self, **options):
        """
        Solves the given model.  Currently, takes as options only the
        various presolve options.  For example, to turn on
        presolve_rows, pass presolve_rows=True as one of the
        arguments.
        
        Available presolve options are:

        presolve_none, presolve_rows, presolve_cols, presolve_lindep,
        presolve_aggregate, presolve_sparser, presolve_sos,
        presolve_reducemip, presolve_knapsack, presolve_elimeq2,
        presolve_impliedfree, presolve_reducegcd, presolve_probefix,
        presolve_probereduce, presolve_rowdominate,
        presolve_coldominate, presolve_mergerows, presolve_impliedslk,
        presolve_colfixdual, presolve_bounds, presolve_duals,
        presolve_sensduals.

        Note: in my experience, the best all-around combination seems
        to be presolve_rows, presolve_cols, presolve_sparser and
        presolve_lindep, but this might be quite problem dependent.
        """

        # Get the current options dict
        cdef dict option_dict = self.getOptions().copy()
        option_dict.update(options)

        # See if there are any flags having to do with the presolve
        cdef str k
        cdef unsigned long n
        cdef unsigned long presolve = 0

        for k, n in _presolve_flags.iteritems():
            if k in option_dict and option_dict[k]:
                presolve += n
        
        set_presolve(self.lp, presolve, 100)

        cdef int ret = solve(self.lp)
        
        if ret == 0:
            return
        elif ret == -2:
            # NOMEMORY (-2)  	Out of memory
            raise MemoryError("LP Solver out of memory.")
        elif ret == 1:
            # SUBOPTIMAL (1) The model is sub-optimal. Only happens if
            # there are integer variables and there is already an
            # integer solution found.  The solution is not guaranteed
            # the most optimal one.

            # A timeout occured (set via set_timeout or with the -timeout option in lp_solve)
            # set_break_at_first was called so that the first found integer solution is found (-f option in lp_solve)
            # set_break_at_value was called so that when integer solution is found 
            #   that is better than the specified value that it stops (-o option in lp_solve)
            # set_mip_gap was called (-g/-ga/-gr options in lp_solve) to specify a MIP gap
            # An abort function is installed (put_abortfunc) and this function returned TRUE
            # At some point not enough memory could not be allocated 

            warnings.warn("Solver solution suboptimal")
        elif ret == 2:
            # INFEASIBLE (2) 	The model is infeasible
            raise LPSolveException("Error 2: Model infeasible")
        elif ret == 3:
            # UNBOUNDED (3) 	The model is unbounded
            raise LPSolveException("Error 3: Model unbounded")
        elif ret == 4:
            # DEGENERATE (4) 	The model is degenerative
            raise LPSolveException("Error 4: Model degenerate")
        elif ret == 5:
            # NUMFAILURE (5) 	Numerical failure encountered
            raise LPSolveException("Error 5: Numerical failure encountered")
        elif ret == 6:
            # USERABORT (6) 	The abort routine returned TRUE. See put_abortfunc
            raise LPSolveException("Error 6: Solver aborted")
        elif ret == 7:
            # TIMEOUT (7) 	A timeout occurred. Indicates timeout was set via set_timeout
            raise LPSolveException("Error 7: Timeout Occurred.")
        elif ret == 9:
            # PRESOLVED (9) The model could be solved by
            # presolve. This can only happen if presolve is active via
            # set_presolve
            return
        elif ret == 10:
            # PROCFAIL (10) 	The B&B routine failed
            raise LPSolveException("Error 10: The B&B routine failed")
        elif ret == 11:
            # PROCBREAK (11) The B&B was stopped because of a
            # break-at-first (see set_break_at_first) or a
            # break-at-value (see set_break_at_value)
            raise LPSolveException("Error 11: B&B Stopped.")
        elif ret == 12:
            # FEASFOUND (12) 	A feasible B&B solution was found
            return 
        elif ret == 13:
             # NOFEASFOUND (13) 	No feasible B&B solution found
            raise LPSolveException("Error 13: No feasible B&B solution found")


    cpdef ar getFinalVariables(self):
        """
        Returns the final values of the variables in the constraints.
        """

        cdef size_t n = get_Ncolumns(self.lp)

        self._resize_real_buffer(n)

        if not get_variables(self.lp, self.rbuf):
            raise LPSolveException("Error retrieving final variables")
        
        cdef ar[double, mode="c"] v = empty(n)
        cdef size_t i
        
        for i from 0 <= i < n:
            v[i] = self.rbuf[i]

        return v

    cpdef real getObjectiveValue(self):
        """
        Returns the value of the objective function of the LP.
        """

        return get_objective(self.lp)

    cpdef print(self):
        print_lp(self.lp)


    ##################################################
    # Now the lower level stuff

    cdef _check_full_sizing(self, ar a):
        cdef size_t nr = get_Ncolumns(self.lp)
    
        if a.shape[0] != nr:
            raise LPSolveException("Row size (%d) does not equal the current number of columns (%d)."
                                   % (a.shape[0], nr))

    ##############################
    # Buffer control 

    cdef _copy_into_real_buffer(self, ar a):
    
        self._resize_real_buffer(a.shape[0])

        if a.dtype == np.float32:
            self._copy_into_real_buffer_float32(a)
        elif a.dtype == np.float64:
            self._copy_into_real_buffer_float64(a)
        else:
            self._copy_into_real_buffer_float64(np.asarray(a, dtype=np.float64))

    cdef _copy_into_real_buffer_float32(self, ar a_float32_o):
        cdef size_t i
        cdef ar[float32_t] a_float32 = a_float32_o

        with cython.boundscheck(False):
            for i from 0 <= i < a_float32.shape[0]:
                self.rbuf[i] = a_float32[i]

    cdef _copy_into_real_buffer_float64(self, ar a_float64_o):
        cdef size_t i
        cdef ar[float64_t] a_float64 = a_float64_o

        with cython.boundscheck(False):
            for i from 0 <= i < a_float64.shape[0]:
                self.rbuf[i] = a_float64[i]
        
    
    cdef _copy_into_int_buffer(self, ar a):

        self._resize_int_buffer(a.shape[0])

        if a.dtype is np.int32:
            self._copy_into_int_buffer_int32(a)
        elif a.dtype is np.uint32:
            self._copy_into_int_buffer_uint32(a)
        elif a.dtype is np.int64:
            self._copy_into_int_buffer_int64(a)
        elif a.dtype is np.uint64:
            self._copy_into_int_buffer_uint64(a)
        else:
            self._copy_into_int_buffer_uint32(np.asarray(a, dtype=np.uint32))

    cdef _copy_into_int_buffer_int32(self, ar a_o):

        cdef ar[int32_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i]

    cdef _copy_into_int_buffer_uint32(self, ar a_o):

        cdef ar[uint32_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i]

    cdef _copy_into_int_buffer_int64(self, ar a_o):

        cdef ar[int64_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i]

    cdef _copy_into_int_buffer_uint64(self, ar a_o):

        cdef ar[uint64_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i]


    cdef _copy_into_int_buffer_shift(self, ar a):

        self._resize_int_buffer(a.shape[0])

        if a.dtype is np.int32:
            self._copy_into_int_buffer_int32_shift(a)
        elif a.dtype is np.uint32:
            self._copy_into_int_buffer_uint32_shift(a)
        elif a.dtype is np.int64:
            self._copy_into_int_buffer_int64_shift(a)
        elif a.dtype is np.uint64:
            self._copy_into_int_buffer_uint64_shift(a)
        else:
            self._copy_into_int_buffer_uint32_shift(np.asarray(a, dtype=np.uint32))

    cdef _copy_into_int_buffer_int32_shift(self, ar a_o):

        cdef ar[int32_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i] + 1

    cdef _copy_into_int_buffer_uint32_shift(self, ar a_o):

        cdef ar[uint32_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i] + 1

    cdef _copy_into_int_buffer_int64_shift(self, ar a_o):

        cdef ar[int64_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i] + 1

    cdef _copy_into_int_buffer_uint64_shift(self, ar a_o):

        cdef ar[uint64_t] a = a_o
        cdef size_t i

        with cython.boundscheck(False):
            for i from 0 <= i < a.shape[0]:
                self.intbuf[i] = a[i] + 1

    cdef void _resize_real_buffer(self, size_t n):
        if self.rbuf_allocated == NULL:
            self.rbuf_allocated = <real*>malloc( (n+1)*sizeof(real) )
        elif self.rbuf_size < n:
            self.rbuf_allocated = <real*>realloc(self.rbuf_allocated, (n+1)*sizeof(real))
        
        self.rbuf = self.rbuf_allocated + 1
        self.rbuf_size = n
        

    cdef void _resize_int_buffer(self, size_t n):
        if self.intbuf_allocated == NULL:
            self.intbuf_allocated = <int*>malloc( (n+1)*sizeof(int) )
        elif self.intbuf_size < n:
            self.intbuf_allocated = <int*>realloc(self.intbuf_allocated, (n+1)*sizeof(int))
        
        self.intbuf = self.intbuf_allocated + 1
        self.intbuf_size = n
        

