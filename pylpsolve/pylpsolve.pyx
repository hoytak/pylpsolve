from numpy cimport ndarray as ar, \
    int_t, uint_t, int32_t, uint32_t, int64_t, uint64_t, float_t

cimport cython

from numpy import int32,uint32,int64, uint64, float32, float64,\
    uint, empty, ones, zeros, uint, arange, isscalar, amax, amin, \
    ndarray, array, asarray, isfinite, argsort

from typeconfig import npint, npuint, npfloat

import warnings
import optionlookup

ctypedef unsigned char ecode
ctypedef double real

############################################################
# See if we can support sparse matrics in specifying constraints 

# cdef bint sparse_supported

# try:
#     import scipy.sparse as sp
#     sparse_supported = True
# except ImportError:
#     sp = None
#     sparse_supported = False

# cdef object issparse = sp.issparse if sparse_supported else None
# cdef object spfind   = sp.find     if sparse_supported else None


##############################
# This is how it should work
#from typechecks cimport *

#This works

############################################################
# Miscilaneous utility functions for resolving types

from types import IntType, LongType, FloatType
from numpy import isscalar

cdef inline isnumeric(v):
    t_v = type(v)

    global IntType
    
    if (t_v is IntType
        or t_v is LongType
        or t_v is FloatType):
        return True
    else:
        return isscalar(v)

cdef inline isposint(v):
    t_v = type(v)

    if (t_v is IntType or t_v is LongType) and v >= 0:
        return True


cdef inline bint issize(v):
    t_v = type(v)

    if (t_v is IntType or t_v is LongType) and v >= 1:
        return True

cdef inline bint istuplelist(list l):

    for t in l:
        if type(t) is not tuple or len(<tuple>t) != 2:
            return False

    return True
        
cdef inline bint isnumericlist(list l):
    for t in l:
        if not isnumeric(t):
            return False

    return True

cdef inline bint is2dlist(list l):
    cdef int length = -1

    for ll in l:
        if not isnumericlist(ll):
            return False
        if length == -1:
            length = len(<list>ll)
        else:
            if length != len(<list>ll):
                return False
    else:
        return True

cdef inline bint isposintlist(list l):
    for t in l:
        if not isposint(t):
            return False

    return True

cdef inline bint isfullindexarray(ar a_o, size_t n_cols):
    cdef ar[int] a = a_o

    if a.shape[0] != n_cols:
        return False

    cdef ar[int, mode="c"] count = zeros(n_cols)

    cdef size_t i
    cdef int v 

    with cython.boundscheck(False):
        for 0 <= i < n_cols:
            v = a[i]
            if v < 0 or v >= n_cols:
                return False

            if count[v] != 0:
                return False

            count[v] = 1


######################################################################
# A few early binding things

cdef dict default_options = optionlookup._default_options
cdef dict presolve_flags  = optionlookup._presolve_flags
cdef dict pricer_lookup   = optionlookup._pricer_lookup
cdef dict pricer_flags    = optionlookup._pricer_flags
cdef dict scaling_lookup  = optionlookup._scaling_lookup
cdef dict scaling_flags   = optionlookup._scaling_flags

cdef double infty = 1e30
cdef ar pos_arinfty = array([infty], npfloat)
cdef ar neg_arinfty = array([-infty], npfloat)
cdef tuple range_tuple_size = (1, 2)


######################################################################
# Forward declarations

cdef struct _Constraint

######################################################################
# Exceptions

class LPSolveException(Exception): pass

######################################################################
# Constraint types

DEF constraint_free    = 0
DEF constraint_leq     = 1
DEF constraint_geq     = 2
DEF constraint_equal   = 3
DEF constraint_in      = 4

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

cdef dict _constraint_type_rev_map = {
    constraint_leq    : "<=",
    constraint_equal  : "=",
    constraint_geq    : ">=",
    constraint_in     : "in"}

cdef inline str getReverseCType(int ctype):
    return _constraint_type_rev_map[ctype]

cdef dict _constraint_map = dict(_constraint_type_list)
cdef str _valid_constraint_identifiers = \
    ','.join(["'%s'" % cid for cid,ct in _constraint_type_list])


cdef inline bint isSimpleCType(int ctype):
    return ctype in [constraint_leq, constraint_geq, constraint_equal]

cdef inline getCType(str ctypestr):  # no return typing to allow exceptions
    try:
        return _constraint_map[ctypestr]
    except KeyError:
        raise ValueError("Constraint type '%s' not recognized." % ctypestr)


######################################################################
# External imports

cdef extern from "lpsolve/lp_lib.h":
    ctypedef void lprec

    lprec* make_lp(int rows, int cols)
    void delete_lp(lprec*)

    ecode resize_lp(lprec *lp, int rows, int columns)

    ecode set_obj_fn(lprec*, real* row)
    ecode set_obj_fnex(lprec*, int count, real*row, int *colno)
    void set_maxim(lprec *lp)
    void set_minim(lprec *lp)

    ecode add_constraint(lprec*, real* row, int ctype, real rh)
    ecode add_constraintex(lprec*, int count, real* row, int *colno, int ctype, real rh)

    ecode set_rowex(lprec *lp, int row_no, int count, real *row, int *colno)
    ecode set_constr_type(lprec *lp, int row, int con_type)
    ecode set_rh(lprec *lp, int row, real value)
    ecode set_rh_range(lprec *lp, int row, real deltavalue)

    ecode set_add_rowmode(lprec*, unsigned char turn_on)

    # Variable bounds
    ecode set_lowbo(lprec*, int column, real value)
    ecode set_upbo(lprec*, int column, real value)
    real get_lowbo(lprec*, int column)
    real get_upbo(lprec*, int column)
    real get_infinite(lprec*)
    ecode set_bounds(lprec*, int column, real lower, real upper)
    ecode set_unbounded(lprec *lp, int column)

    # Row bounds
    ecode set_rowex(lprec *lp, int row_no, int count, real *row, int *colno)
    ecode set_row(lprec *lp, int row_no, real *row)
    ecode set_constr_type(lprec *lp, int row, int con_type)
    ecode set_rh(lprec *lp, int row, real value)
    ecode set_rh_range(lprec *lp, int row, real deltavalue)

    # presolve, scaling
    void set_presolve(lprec*, int do_presolve, int maxloops)
    void set_scaling(lprec *, int scalemode)

    real get_var_primalresult(lprec *lp, int index)

    # Basis stuff
    ecode guess_basis(lprec *lp, real *guessvector, int *basisvector)
    ecode set_basis(lprec *lp, int *bascolumn, bint nonbasic)
    ecode get_basis(lprec *lp, int *bascolumn, unsigned char nonbasic)

    int get_Ncolumns(lprec*)
    int get_Nrows(lprec*)

    ecode get_ptr_variables(lprec*, real **var)
    real get_objective(lprec*)

    # pivoting
    void set_pivoting(lprec*, int rule)
    void set_sense(lprec *lp, bint maximize)
    
    int solve(lprec *lp)

    int print_lp(lprec *lp)
    
    void set_verbose(lprec*, int)

    # Retrieving statistics
    long long get_total_iter(lprec *lp)


############################################################
# Structs for buffering the constraints

cdef extern from "Python.h":
    void* malloc "PyMem_Malloc"(size_t n)
    void* realloc "PyMem_Realloc"(void *p, size_t n)
    void free "PyMem_Free"(void *p)

    void Py_INCREF(object)


cdef extern from "stdlib.h":
    void memset(void *p, char value, size_t n)

DEF nIterations = 0

cdef dict info_lookup = {
    "iterations" : nIterations}

################################################################################
# Now the full class

DEF m_NewModel      = 0
DEF m_UpdatingModel = 1

# This is the size of the constraint buffer blocks
DEF cStructBufferSize = 128  

cdef class LPSolve(object):
    """
    The class wrapping the lp_solve api 
    """

    cdef lprec *lp
    cdef size_t n_columns, n_rows

    # Option dict
    cdef dict options

    # constraints
    cdef _Constraint **_c_buffer
    cdef size_t current_c_buffer_size

    # Variable bounds
    cdef list _lower_bound_keys, _lower_bound_values
    cdef size_t _lower_bound_count

    cdef list _upper_bound_keys, _upper_bound_values
    cdef size_t _upper_bound_count
    
    # The objective function
    cdef list _obj_func_keys
    cdef list _obj_func_values
    cdef bint _obj_func_specified
    cdef size_t _obj_func_n_vals_count 
    cdef bint _maximize_mode

    # Methods relating to named variable group
    cdef dict _named_index_blocks
    cdef bint _nonindexed_blocks_present, _user_warned_about_nonindexed_blocks

    # Variables relating to post-solve aspects
    cdef ar final_variables

    def __cinit__(self):

        self.lp = NULL
        self.options = default_options.copy()

        self.current_c_buffer_size = 0
        self._c_buffer = NULL
        
        self._clear(False)

    def __dealloc__(self):
        self.clearConstraintBuffers()
        
        if self.lp != NULL:
            delete_lp(self.lp)

    def clear(self):
        self._clear(False)

    cdef _clear(self, bint restartable_mode):
        # If restartable mode, only delete all the temporary stuff
        # but still allow the user to tweak the model and resume the LP

        self._lower_bound_keys   = []
        self._lower_bound_values = []
        self._lower_bound_count  = 0

        self._upper_bound_keys   = []
        self._upper_bound_values = []
        self._upper_bound_count  = 0

        self._obj_func_keys   = []
        self._obj_func_values = []
        self._obj_func_specified = False
        self._obj_func_n_vals_count = 0

        self.clearConstraintBuffers()

        self.final_variables = None

        if not restartable_mode:
            if self.lp != NULL:
                delete_lp(self.lp)
                self.lp = NULL

            # For the minimize default
            self._maximize_mode = False

            self.n_rows = 0
            self.n_columns = 0
            
            self._named_index_blocks = {}
            self._user_warned_about_nonindexed_blocks = False
            self._nonindexed_blocks_present = False


    ############################################################
    # Methods concerned with getting and setting the options

    cpdef dict getOptionDict(self):
        """
        Returns a copy of the dictionary containing all the current
        options.
        """

        return self.options.copy()

    def getOption(self, str option):
        """
        Returns the current default value of option `option`.
        """

        try:
            return self.options[option]
        except KeyError:
            raise ValueError("Option '%s' not valid." % str(option))

    def setOption(self, *args, **kw_options):
        """
        Sets the current default value for an option(s).  Options may
        either be passed in pairs as arguments or as keyword
        arguments.  Thus the following is valid::

          lp.setOption("pricer", "devex", presolve_rows = True)

        This sets the pricer option to "devex" and presolve_rows to
        True.


        Presolve options
        ==================================================
        
        Presolve is off by default, as it prevents specifiying a basis
        or initial guess or modifying the constraint matrix
        afterwards.  To turn it on, set any of the options below to
        True.

        Available presolve options are::

          presolve_rows:
            Presolve rows.

          presolve_cols:
            Presolve columns.

          presolve_lindep:
            Eliminate linearly dependent rows.

          presolve_sos:
            Convert constraints to SOSes (only SOS1 handled).

          presolve_reducemip:
            If the phase 1 solution process finds that a constraint is
            redundant then this constraint is deleted. This is no
            longer active since it is very rare that this is
            effective, and also that it adds code complications and
            delayed presolve effects that are not captured properly.

          presolve_knapsack:
            Simplification of knapsack-type constraints through
            addition of an extra variable, which also helps bound the
            OF.

          presolve_elimeq2:
            Direct substitution of one variable in 2-element equality
            constraints; this requires changes to the constraint
            matrix.

          presolve_impliedfree:
            Identify implied free variables (releasing their explicit
            bounds).

          presolve_reducegcd: 
            Reduce (tighten) coefficients in integer models based on
            GCD argument.

          presolve_probefix:
            Attempt to fix binary variables at one of their bounds.

          presolve_probereduce:
            Attempt to reduce coefficients in binary models.

          presolve_rowdominate: 
            Idenfify and delete qualifying constraints that are
            dominated by others, also fixes variables at a bound.

          presolve_coldominate:
            Deletes variables (mainly binary), that are dominated by
            others (only one can be non-zero).

          presolve_mergerows:
            Merges neighboring >= or <= constraints when the vectors
            are otherwise relatively identical into a single ranged
            constraint.

          presolve_impliedslk:
            Converts qualifying equalities to inequalities by
            converting a column singleton variable to slack. The
            routine also detects implicit duplicate slacks from
            inequality constraints, fixes and removes the redundant
            variable. This latter removal also tends to reduce the
            risk of degeneracy. The combined function of this option
            can have a dramatic simplifying effect on some models.

          presolve_colfixdual: 
            Variable fixing and removal based on considering signs of
            the associated dual constraint.

          presolve_bounds:
            Does bound tightening based on full-row constraint
            information. This can assist in tightening the OF bound,
            eliminate variables and constraints. At the end of
            presolve, it is checked if any variables can be deemed
            free, thereby reducing any chance that degeneracy is
            introduced via this presolve option..

          presolve_duals:
            Calculate duals.

          presolve_sensduals:
            Calculate sensitivity if there are integer variables.

        Presolve options are turned on (or disabled, if turned on
        previously with setOption) by passing ``presolve_x = True``.


        Pricing and Pivoting Options
        ==================================================
        
        The main pricing options can take the following values::

          "firstindex":
            Select first available pivot.

          "dantzig": 
            Select according to Dantzig.

          "devex":
             Devex pricing from Paula Harris.
             
          "steepestedge":
             Steepest Edge.

        
        Addition pricer options, which may be set using True::
    
          price_primalfallback: 
            In case of Steepest Edge, fall back to DEVEX in primal.

          price_multiple:	
            Preliminary implementation of the multiple pricing
            scheme. This means that attractive candidate entering
            columns from one iteration may be used in the subsequent
            iteration, avoiding full updating of reduced costs.  In
            the current implementation, lp_solve only reuses the 2nd
            best entering column alternative.

          price_partial:	
            Enable partial pricing.

          price_adaptive:	
            Temporarily use alternative strategy if cycling is detected.

          price_randomize:	
            Adds a small randomization effect to the selected pricer.

          price_autopartial:	
            Indicates automatic detection of segmented/staged/blocked
            models. It refers to partial pricing rather than full
            pricing. With full pricing, all non-basic columns are
            scanned, but with partial pricing only a subset is scanned
            for every iteration. This can speed up several models.

          price_loopleft:	
            Scan entering/leaving columns left rather than right.

          price_loopalternate:	
            Scan entering/leaving columns alternatingly left/right.

          price_harristwopass:	
            Use Harris' primal pivot logic rather than the default.

          price_truenorminit:	
            Use true norms for Devex and Steepest Edge initializations.

        Scaling Options
        ========================================

        There's a primary scaling mode plus additional flags may be
        set.  The scaling mode can influence numerical stability
        considerably.  It is advisable to always use some sort of
        scaling.  

        The available scale modes are set using the ``scale_mode``
        option, which can take the following values:

          "none":
            No scaling used.

          "extreme":
            Scale to convergence using largest absolute value.

          "range":
            Scale based on the simple numerical range.

          "mean":
            Numerical range-based scaling.

          "geometric":
            Geometric scaling (default).

          "curtisreid":
            Curtis-reid scaling.


        In addition, the following options may be passed as options
        (e.g. ``scale_integers = True``).

          scale_logarithmic:
	    Scale to convergence using logarithmic mean of all values.

          scale_userweight:
            User can specify scalars (not implemented).

          scale_power2:
            Also do Power scaling (off by default).

          scale_equilibrate: 
            Make sure that no scaled number is above 1 (on by default).

          scale_integers:
            Scale integer variables (off by default).

          scale_dynupdate: 
            It has always been so that scaling is done only once on
            the original model. If a solve is done again (most
            probably after changing some data in the model), the
            scaling factors aren't computed again. The scalars of the
            original model are used. This is not always good,
            especially if the data has changed considerably. If
            scale_dynupdate is True, the scaling factors are
            recomputed also when a restart is done. Note that they are
            then always recalculated with each solve, even when no
            change was made to the model, or a change that doesn't
            influence the scaling factors like changing the RHS (Right
            Hand Side) values or the bounds/ranges. This can influence
            performance. It is up to you to decide if scaling factors
            must be recomputed or not for a new solve, but by default
            it still isn't so. It is possible to set/unset this flag
            at each next solve and it is even allowed to choose a new
            scaling algorithm between each solve. Note that the
            scaling done by the scale_dynupdate is incremental and the
            resulting scalars are typically different from scalars
            recomputed from scratch. 

          scale_rowsonly:
            Scale only rows.
            
          scale_colsonly:
            Scale only columns.

        Other miscilaneous options::
        ========================================


          verbosity:
            Sets the verbosity level of printed information.  Default
            is 1, which only prints errors; levels 1-5 are available,
            with 6 being the most verbose.

        """

        if len(args) % 2 != 0:
            raise ValueError("Non keyword arguments must be in option, value pairs.")

        for i, (n, v) in enumerate(zip(args[::2], args[1::2])):
            if not isinstance(n, str):
                raise TypeError("Argument %d not str." % 2*i)
            if n not in self.options:
                raise ValueError("Option '%s' not valid." % n)

            if n not in kw_options:
                kw_options[n] = v

        # Check everything, so we reject as a group if one is bad
        for k in kw_options.iterkeys():
            if k not in self.options:
                raise ValueError("Option '%s' not valid." % k)

        # All pass, set them
        self.options.update(kw_options)


    ############################################################
    # Methods relating to variable indices

    def getVariables(self, a1, a2 = None):
        """
        Returns a new or current block of 'size' variable indices to
        be used in the current LP.  The return value is a 2-tuple,
        ``(low_index, high_index)``, such that
        ``high_index-low_index=size``  
        
        If `name` is not None, then other functions
        (e.g. addConstraint) can accept this accept this string as the
        index argument.

        # Put in use cases 

        # Blah

        """
        
        cdef str name
        cdef size_t size

        cdef ar[size_t, ndim=2, mode="c"] B 

        if a2 is None:
            if type(a1) is str:
                name = <str>a1

                if name not in self._named_index_blocks:
                    raise ValueError("Block '%s' must be declared with size on first call." % name)

                B = self._getVariableIndexBlock(0, name)
            elif issize(a1):
                B = self._getVariableIndexBlock(<size_t>a1, None)
            else:
                raise ValueError("First argument must either be name of variable block or size (>= 1) of new block.")
        else:
            if type(a2) is str and issize(a1):
                name = a2
                size = a1
            elif type(a1) is str and issize(a2):
                name = a1
                size = a2
            else:
                raise ValueError("getVariables() arguments must be a block name/size, size, or name of prexisting block.")

            B = self._getVariableIndexBlock(size, name)
            
        assert B.shape[0] == 1
        assert B.shape[1] == 2

        return (B[0,0], B[0,1])

    cdef ar _makeVarIdxRange(self, size_t lower, size_t upper):
        cdef ar[size_t, ndim=2, mode="c"] B = empty(range_tuple_size, npuint)
        
        B[0,0] = lower
        B[0,1] = upper

        assert self._indexBlockSize(B) == upper - lower

        return B

    cdef ar _getVariableIndexBlock(self, size_t size, str name):
        # size = 0 denotes no checking
    
        cdef ar[size_t, ndim=2,mode="c"] idx

        if name is None:
            if size == 0:
                raise ValueError("Size of variable block must be >= 1")

            idx = self._makeVarIdxRange(self.n_columns, self.n_columns + size)
            self.checkColumnCount(self.n_columns + size, True)
            return idx

        else:
            if name in self._named_index_blocks:
                idx = self._named_index_blocks[name]
                if size not in  [0,1] and idx[0,1] - idx[0,0] != size:
                    raise ValueError("Requested size (%d) does not match previous size (%d) of index block '%s'"
                                     % (size, idx[0,1] - idx[0,0], name))
                return idx

            else:
                if size == 0:
                    raise ValueError("Size of variable block must be >= 1")

                idx = self._makeVarIdxRange(self.n_columns, self.n_columns + size)
                self.checkColumnCount(self.n_columns + size, True)
                self._named_index_blocks[name] = idx
                return idx

    cdef bint _isCurrentIndexBlock(self, idx_block):
        if type(idx_block) is str:
            return idx_block in self._named_index_blocks
        else:
            return True

    ########################################
    # For internally resolving things
    cdef ar _resolveIdxBlock(self, idx, size_t n):
    
        if idx is None:
            self.checkColumnCount(n, True)
            return None

        cdef ar ar_idx
        cdef ar[int, mode="c"] ai
        cdef long v_idx
        cdef bint error = False
        cdef tuple t
        cdef size_t a, b

        if type(idx) is str:
            return self._getVariableIndexBlock(n, <str>idx)

        elif type(idx) is ndarray:
            ar_idx = idx

            if ar_idx.ndim != 1:
                raise ValueError("Index array must be 1d vector.")

            if n not in [ar_idx.shape[0], 1]:
                raise ValueError("Number of values (%d) must equal number of indices (%d) or 1." 
                                 % (n, ar_idx.shape[0]))

            # Need to freeze the data
            if ar_idx.dtype is npint:
                ai = ar_idx.copy()
            else:
                ai = array(ar_idx, npint)
                
                if (ai != ar_idx).any():
                    raise ValueError("Unable to convert index array to nonnegative integers")
            
            if amin(ai) < 0:
                raise ValueError("Unable to convert index array to nonnegative integers")

            self.checkColumnCount(amax(ai) + 1, True)
            
            return ai

        elif type(idx) is tuple:
            t = <tuple>idx
            self._validateIndexTuple(t)

            t0 = <size_t>t[0]
            t1 = <size_t>t[1]

            if n not in [t1 - t0, 1]:
                raise ValueError("Number of values (%d) does not equal size of index range (%d) or 1." % (n, t1 - t0))

            self.checkColumnCount(t1, True)
            
            return self._makeVarIdxRange(t0,t1)

        elif type(idx) is list:
            try:
                ar_idx = array(idx,dtype=npint)
            except Exception, e:
                raise ValueError("Error converting index list to 1d integer array: %s" % str(e))
            
            if ar_idx.ndim != 1:
                raise ValueError("Error interpreting index list: Not 1 dimensional.")
            if n not in [ar_idx.shape[0], 1]:
                raise ValueError("Number of values (%d) must equal number of indices (%d) or 1." 
                                 % (n, ar_idx.shape[0]))

            if amin(ar_idx) < 0:
                raise ValueError("Negative indices not allowed.")
            
            self.checkColumnCount(amax(ar_idx) + 1, True)

            return ar_idx

        elif isnumeric(idx):
            v_idx = <long>idx
            if v_idx < 0 or v_idx != idx:
                raise ValueError("%s not valid as nonnegative index. " % str(idx))

            self.checkColumnCount(v_idx, True)
            ar_idx = empty(1, npint)
            ar_idx[0] = idx
            return ar_idx

        else:
            raise TypeError("Type of index (%s) not recognized; must be scalar, list, tuple, str, or array." % type(idx))

    cdef size_t _indexBlockSize(self, ar idxblock):
        if idxblock.ndim == 2:
            assert idxblock.shape[0] == 1
            assert idxblock.shape[1] == 2
            return ((<size_t*>idxblock.data)[1] - ((<size_t*>idxblock.data)[0]))
        elif idxblock.ndim == 1:
            return idxblock.shape[0]

    cdef bint _isIndexBlock(self, ar idxblock):
        return idxblock.ndim == 2

    cdef size_t _indexBlockLower(self, ar idxblock):
        assert idxblock.ndim == 2
        assert idxblock.shape[0] == 1
        assert idxblock.shape[1] == 2
        assert (<size_t*>idxblock.data)[0] == idxblock[0,0]
        return (<size_t*>idxblock.data)[0]

    cdef size_t _indexBlockUpper(self, ar idxblock):
        assert idxblock.ndim == 2
        assert idxblock.shape[0] == 1
        assert idxblock.shape[1] == 2
        assert (<size_t*>idxblock.data)[1] == idxblock[0,1]
        return (<size_t*>idxblock.data)[1]

    cdef _validateIndexTuple(self, tuple t):
        cdef int v_idx, w_idx
        cdef bint error = False
    
        if len(t) != 2: 
            self._raiseTupleValidationError(t)

        t1, t2 = t

        if not isnumeric(t1) or not isnumeric(t2):
            self._raiseTupleValidationError(t)
            
        v_idx, w_idx = t1, t2
            
        if v_idx < 0 or w_idx < 0:
            self._raiseTupleValidationError(t)

        if not w_idx > v_idx:
            self._raiseTupleValidationError(t)
    
    cdef _raiseTupleValidationError(self, tuple t):
        raise ValueError("Index tuples must be of form (lower_index, upper_index), with nonnegative indices ( %s )." % str(t))

    cdef ar _resolveValues(self, v, bint ensure_1d):
        cdef ar ret

        if type(v) is ndarray or type(v) is list or type(v) is tuple: 
            ret = asarray(v, npfloat)
        elif isnumeric(v):
            ret = empty(1, npfloat)
            ret[0] = v
        else:
            raise TypeError("Type of values (%s) not recognized." % str(type(v)))

        if ret.ndim == 1:
            return ret
        elif ret.ndim == 2:
            if ensure_1d:
                raise ValueError("Only 1d arrays allowed for values here.")
            else:
                return ret
        else:
            raise ValueError("Dimension of values array must be either 1 or 2.")
        

    ############################################################
    # Bookkeeping for the column counts

    cdef checkColumnCount(self, size_t requested_size, bint indexed):

        if requested_size < self.n_columns:
            if not indexed:
                self._warnAboutNonindexedBlocks()

        elif requested_size > self.n_columns:
            if self._nonindexed_blocks_present:
                self._warnAboutNonindexedBlocks()

            self.n_columns = requested_size

        if not indexed:
            self._nonindexed_blocks_present = True


    cdef _warnAboutNonindexedBlocks(self):
        if not self._user_warned_about_nonindexed_blocks:
            #warnings.warn("Non-indexed variable block present which does not span columns; "
            #              "setting to lowest-indexed columns.")
            raise Exception("Non-indexed variable block present which does not span columns; "
                            "setting to lowest-indexed columns.")
            self._user_warned_about_nonindexed_blocks = True
            


    ############################################################
    # Methods dealing with constraints

    cpdef addConstraint(self, coefficients, str ctypestr, rhs):
        """        
        Adds a constraint, or set of constraints to the lp.

        `coefficients` may be either a single array, a dictionary, or
        a 2-tuple with the form (index block, value array).  In the
        case of a single array, it must be either 1 or 2 dimensional
        and have the same length/number of columns as the number of
        columns in the lp.  If it is 2 dimensions, each row
        corresponds to a constraint, and `rhs` must be a vector with a
        coefficient for each row.  In this case, ctype must
        be the same for all rows.

        If it is a dictionary, the keys of the dictionary are the
        indices of the non-zero values which are given by the
        corresponding values.  

        If it is a (index array, value array) pair, the corresponding
        pairs have the same behavior as in the dictionary case.

        `ctype` is a string determining the type of
        inequality or equality in the constraint.  The following are
        all valid identifiers: '<', '<=', '=<', 'lt', 'leq'; '>',
        '>=', '=>', 'gt', 'geq', '=', '==', 'equal', and 'eq'. 
        """
                
        cdef bint is_list_sequence, is_numerical_sequence
        cdef tuple t_coeff

        cdef int ctype = getCType(ctypestr)

        # What we do depends on the type
        coefftype = type(coefficients)

        if coefftype is tuple:

            t_coeff = <tuple>coefficients

            # Make sure it's split into index, value pair
            if len(t_coeff) != 2:
                raise ValueError("coefficients should be either a single array, "
                                 "a dictionary, or a 2-tuple with (index block, values)")

            return self._addConstraintArray(t_coeff[0], t_coeff[1], ctype, rhs)

        elif coefftype is ndarray:
            return self._addConstraintArray(None, coefficients, ctype, rhs)

        elif coefftype is list:
            # Two possible ways to interpret a list; as a sequence of
            # tuples or as a representation of an array; if it is a
            # sequence of 2-tuples, intepret it this way 
            

            # Test and see if it's a list of sequences or numerical list
            is_list_sequence = istuplelist(<list>coefficients)

            if is_list_sequence: 
                is_numerical_sequence = False
            else:
                is_numerical_sequence = isnumericlist(<list>coefficients)
            
            if is_list_sequence:
                return self._addConstraintTupleList(<list>coefficients, ctype, rhs)

            elif is_numerical_sequence:
                return self._addConstraintArray(None, coefficients, ctype, rhs)

            else:

                # try interpreting it as an array

                A = self._attemptArrayList(<list>coefficients)

                if A is None:
                    raise ValueError("Coefficient list must containt one of: scalars, 2-tuples, or lists/1d-arrays.")
                else:
                    return self._addConstraintArray(None, A, ctype, rhs)

        elif coefftype is dict:
            return self._addConstraintDict(<dict>coefficients, ctype, rhs)

        elif isnumeric(coefficients):
            return self._addConstraintArray(None, [coefficients], ctype, rhs)

        else:
            raise ValueError("Coefficients must be dict, list, 2-tuple, or array.")


    cdef _addConstraintArray(self, t_idx, t_val, int ctype, rhs):
        # If the values can be interpreted as an array

        cdef bint val_is_scalar = isnumeric(t_val)

        cdef ar A = self._resolveValues(t_val, False)
        cdef size_t i

        # typing for the rhs under range constraints
        cdef ar[double] rhs_a, rhs_b
        cdef bint a_is_scalar, b_is_scalar
        cdef list rhsl, ret_idx_l

        if A.ndim == 1:
            idx = self._resolveIdxBlock(t_idx, A.shape[0])
            return self._addConstraint(idx, A, ctype, rhs)

        elif A.ndim == 2:
            
            # For this, we need some extra work to resolve the 

            idx = self._resolveIdxBlock(t_idx, A.shape[1])
            
            if ctype in [constraint_leq, constraint_geq, constraint_equal]:
                rhs_a = self._resolveValues(rhs, True)

                if rhs_a.shape[0] == 1:
                    return [self._addConstraint(idx, A[i,:], ctype, rhs_a[0])
                            for 0 <= i < A.shape[0]]
                elif rhs_a.shape[0] == A.shape[0]:
                    return [self._addConstraint(idx, A[i,:], ctype, rhs_a[i])
                            for 0 <= i < A.shape[0]]
                else:
                    raise ValueError("Length of right hand side in constraint must be either 1 or match the number of constraints given.")

            elif ctype == constraint_in:
                if not (type(rhs) is list or type(rhs) is tuple) or not len(rhs) == 2:
                    raise ValueError("Range constraints require rhs to be either 2-tuple or 2-list.")
                
                rhs_a = self._resolveValues(rhs[0], True)
                rhs_b = self._resolveValues(rhs[1], True)

                a_is_scalar = (rhs_a.shape[0] == 1)
                b_is_scalar = (rhs_b.shape[0] == 1)

                if not a_is_scalar and rhs_a.shape[0] != A.shape[0]:
                    raise ValueError("Length of lower bound in range constraint must be either 1 or match the number of constraints given.")

                if not b_is_scalar and rhs_b.shape[0] != A.shape[0]:
                    raise ValueError("Length of upper bound in range constraint must be either 1 or match the number of constraints given.")
                    
                ret_idx_l = [None]*A.shape[0]
                rhsl = [None, None]

                for 0 <= i < A.shape[0]:
                    rhsl[0] = rhs_a[0 if a_is_scalar else i]
                    rhsl[1] = rhs_b[0 if b_is_scalar else i]

                    ret_idx_l[i] = self._addConstraint(idx, A[i, :], ctype, rhsl)

                return ret_idx_l
            else:
                assert False
        else:
            assert False


    cdef _addConstraintDict(self, dict d, int ctype, rhs):
        I, V = self._getArrayPairFromDict(d)
        return self._addConstraint(I, V, ctype, rhs)
            
    cdef _addConstraintTupleList(self, list l, int ctype, rhs):
        I, V = self._getArrayPairFromTupleList(l)
        return self._addConstraint(I, V, ctype, rhs)


    ############################################################
    # Convenience functions dealing with common constraint types

    def bindEach(self, indices_1, str ctypestr, indices_2):
        """
        Constrains each variable in `index_group_1` by the
        corresponding variable in `index_group_2` using the ctype
        relationship.  For example::

            lp.bindEach("x", "<=", "y")

        adds constraints to the LP such that each variable in ``x`` is
        less than or equal to the corresponding variable in ``y``.  

        `index_group_1` and `index_group_2` must specify the same
        number of indices, and can be any valid specification for an
        index group.

        Returns the corresponding list of rows.
        """

        ########################################
        # Check to make sure the constraint type is okay for this

        cdef int ctype = getCType(ctypestr)

        if not isSimpleCType(ctype):
            raise ValueError("Constraint type must be <=, =, or >=.")

        ########################################
        # Validate the indices
        
        cdef ar idx_b1, idx_b2
        cdef size_t size

        idx_b1, idx_b2, size = self.validateConvienceIndexPair(indices_1, indices_2)

        ########################################
        # Now loop through and take care of binding the indices

        cdef size_t i

        cdef bint b1_block_mode = self._isIndexBlock(idx_b1)
        cdef bint b2_block_mode = self._isIndexBlock(idx_b2)

        cdef ar[int, mode="c"] idx_1, idx_2
        cdef size_t b1_lb = 50, b2_lb = 50

        if b1_block_mode:
            b1_lb = self._indexBlockLower(idx_b1)
        else:
            idx_1 = idx_b1

        if b2_block_mode:
            b2_lb = self._indexBlockLower(idx_b2)
        else:
            idx_2 = idx_b2

        cdef ar[uint_t, mode="c"] idx = empty(2, npuint)

        cdef ar[double, mode="c"] row = empty(2, npfloat)
        row[0], row[1] = 1, -1

        cdef list ret_row_idx = [None]*size

        for 0 <= i < size:
            idx[0] = (b1_lb + i) if b1_block_mode else idx_1[i]
            idx[1] = (b2_lb + i) if b2_block_mode else idx_2[i]

            ret_row_idx[i] = self._addConstraint(idx, row, ctype, 0)

        return ret_row_idx

    def bindSandwich(self, constrained_indices, sandwich_indices):
        """
        Constrains the absolute value of each variable in
        `constrained_indices` to be less than or equal to the
        corresponding variable in `sandwich_indices`

        `constrained_indices` and `sandwich_indices` must specify the
        same number of indices, and can be any valid specification for
        an index group.

        Returns the corresponding list of rows.
        """

        ########################################
        # Validate the indices
        
        cdef ar idx_b1, idx_b2
        cdef size_t size

        idx_b1, idx_b2, size = self.validateConvienceIndexPair(constrained_indices, sandwich_indices)

        ########################################
        # Now loop through and take care of binding the indices

        cdef size_t i

        cdef bint b1_block_mode = self._isIndexBlock(idx_b1)
        cdef bint b2_block_mode = self._isIndexBlock(idx_b2)

        cdef ar[int, mode="c"] idx_1, idx_2
        cdef size_t b1_lb = 50, b2_lb = 50

        if b1_block_mode:
            b1_lb = self._indexBlockLower(idx_b1)
        else:
            idx_1 = idx_b1

        if b2_block_mode:
            b2_lb = self._indexBlockLower(idx_b2)
        else:
            idx_2 = idx_b2

        cdef ar[uint_t, mode="c"] idx = empty(2, npuint)

        cdef ar[double, mode="c"] row_1 = empty(2, npfloat)
        cdef ar[double, mode="c"] row_2 = empty(2, npfloat)
        
        row_1[0], row_1[1] = 1, -1
        row_2[0], row_2[1] = -1, -1

        cdef list ret_row_idx = [None]*size*2

        for 0 <= i < size:
            idx[0] = (b1_lb + i) if b1_block_mode else idx_1[i]
            idx[1] = (b2_lb + i) if b2_block_mode else idx_2[i]

            ret_row_idx[2*i] = self._addConstraint(idx, row_1, constraint_leq, 0)
            ret_row_idx[2*i+1] = self._addConstraint(idx, row_2, constraint_leq, 0)

        return ret_row_idx




    
    cdef tuple validateConvienceIndexPair(self, indices_1, indices_2):
        # returns tuple of block1, block2, size

        cdef bint idx_b1_is_known = self._isCurrentIndexBlock(indices_1)
        cdef bint idx_b2_is_known = self._isCurrentIndexBlock(indices_2)
        
        cdef ar idx_b1, idx_b2 
        cdef size_t idx_block_1_size, idx_block_2_size

        # Have to choose how to set these to handle the case of one
        # string being implicitly defined

        if not idx_b1_is_known and not idx_b2_is_known:
            raise ValueError("Both index groups implicitly defined.")

        elif idx_b1_is_known and not idx_b2_is_known:
            idx_b1 = self._resolveIdxBlock(indices_1, 1)
            idx_block_1_size = self._indexBlockSize(idx_b1)
            idx_b2 = self._resolveIdxBlock(indices_2, idx_block_1_size)
            idx_block_2_size = self._indexBlockSize(idx_b2)

        elif not idx_b1_is_known and idx_b2_is_known:
            idx_b2 = self._resolveIdxBlock(indices_2, 1)
            idx_block_2_size = self._indexBlockSize(idx_b2)
            idx_b1 = self._resolveIdxBlock(indices_1, idx_block_2_size)
            idx_block_1_size = self._indexBlockSize(idx_b1)

        else:
            idx_b2 = self._resolveIdxBlock(indices_2, 1)
            idx_block_2_size = self._indexBlockSize(idx_b2)
            idx_b1 = self._resolveIdxBlock(indices_1, 1)
            idx_block_1_size = self._indexBlockSize(idx_b1)
            
        if idx_block_1_size != idx_block_2_size:
            raise ValueError("Index blocks have inconsistent sizes.")

        return (idx_b1, idx_b2, idx_block_1_size)

    ############################################################
    # Functions dealing with the constraint

    cdef void _resetObjective(self):
        # Resets the objective function; clearObjective does this and
        # marks it as specified.

        self._obj_func_keys = []
        self._obj_func_values = []
        self._obj_func_n_vals_count = 0
        self._obj_func_specified = False
        
    cpdef clearObjective(self):
        """
        Resets the current objective function.  
        """
        
        self._resetObjective()
        self._obj_func_specified = True

    def setObjective(self, coefficients, mode = None):
        """
        Sets coefficients of the objective function.  Takes
        as arguments 
        """

        # clear the objective (as opposed to addToObjective)
        self.clearObjective()
        self._addToObjective(coefficients)

        self._setMode(mode)


    def addToObjective(self, coefficients, mode = None):
        """
        Just like `setObjective()`, but does not clear the objective
        function first.  Thus this function can be called repeatedly
        to build up different parts of the objective.  Any previously
        specified values are overwritten elementwise.
        """

        self._addToObjective(coefficients)


    cdef _addToObjective(self, coefficients):
        
        # What we do depends on the type
        coefftype = type(coefficients)

        cdef tuple t

        if coefftype is tuple:
            t = <tuple>coefficients
            
            # It's split into index, value pair
            if len(t) != 2:
                raise ValueError("coefficients must be a single array, list,"
                                 "dictionary, or a 2-tuple with (index block, values)")

            idx, val = t

        elif coefftype is ndarray:
            idx, val = None, coefficients

        elif coefftype is list:
            # Two possible ways to interpret a list; as a sequence of
            # tuples or as a representation of an array; if it is a
            # sequence of 2-tuples, intepret it this way 
            
            # Test and see if it's a list of sequences
            is_list_sequence = istuplelist(<list>coefficients)

            # Test and see if it's a list of scalars
            if is_list_sequence: 
                is_numerical_sequence = False
            else:
                is_numerical_sequence = isnumericlist(<list>coefficients)

            if is_list_sequence:
                idx, val = self._getArrayPairFromTupleList(<list>coefficients)
            elif is_numerical_sequence:
                idx, val = None, array(<list>coefficients, npfloat)
            else:
                raise TypeError("Coefficient list must be either list of scalars or list of 2-tuples.")
                    
        elif coefftype is dict:
            idx, val = self._getArrayPairFromDict(<dict>coefficients)

        elif isnumeric(coefficients):
            idx, val = None, [coefficients]

        else:
            raise TypeError("Type of coefficients not recognized; must be dict, list, 2-tuple, or array.")
        
        # debug note: at this point val is ndarray
        self._stackOnInterpretedKeyValuePair(
            self._obj_func_keys, self._obj_func_values, idx, val, &self._obj_func_n_vals_count)

        self._obj_func_specified = True

    cdef applyObjective(self):
        
        assert self.lp != NULL

        cdef ar[int, mode="c"]    I
        cdef ar[double, mode="c"] V
        cdef size_t i 

        if self._maximize_mode:
            set_maxim(self.lp)
        else:
            set_minim(self.lp)
            
        if self._obj_func_specified:

            I, V = self._getArrayPairFromKeyValuePair(
                self._obj_func_keys, 
                self._obj_func_values, 
                self._obj_func_n_vals_count)

            self._resetObjective()

            # necessary for the weird start-at-1 indexing
            for 0 <= i < I.shape[0]: 
                I[i] += 1

            set_obj_fnex(self.lp, I.shape[0], <double*>V.data, <int*>I.data)

    ############################################################
    # Mode stuff

    cpdef setMinimize(self, bint minimize=True):
        """
        If `minimize` is True (default), sets the run mode to minimize
        the objective function.  If `minimize` is false, the run mode
        is set to maximize the objective function.
        """

        self._maximize_mode = not minimize

    cpdef setMaximize(self, bint maximize=True):
        """
        If `minimize` is True (default), sets the run mode to minimize
        the objective function.  If `minimize` is false, the run mode
        is set to maximize the objective function.
        """

        self._maximize_mode = maximize

    cdef _setMode(self, mode):
        if mode == None:
            return
        elif mode == "minimize":
            self.setMinimize()
        elif mode == "maximize":
            self.setMaximize()
        else:
            raise ValueError("mode must be either 'minimize' or 'maximize'")



    ############################################################
    # Methods dealing with variable bounds

    cpdef setUnbounded(self, var):
        """
        Sets the variable `var` to unbounded (default is >=0).  This
        is equivalent to setLowerBound(None), setUpperBound(None)
        """
        
        self.setLowerBound(var, None)
        self.setUpperBound(var, None)

    cpdef setLowerBound(self, var, lb):
        """
        Sets the lower bound of variable(s) `var` to lb.  If lb is None,
        then it sets the lower bound to -Infinity.

        `var` may be a single index, an array, list, or tuple of
        indices, or the name of a block of indices.  If multiple
        indices are specified, then lb may either be a scalar or a
        vector of the same length as the number of indices.
        """

        self._setBound(var, lb, True)

    cpdef setUpperBound(self, var, ub):
        """
        Sets the upper bound of variable idx to ub.  If ub is None,
        then it sets the upper bound to Infinity.

        `var` may be a single index, an array, list, or tuple of
        indices, or the name of a block of indices.  If multiple
        indices are specified, then ub may either be a scalar or a
        vector of the same length as the number of indices.
        """

        self._setBound(var, ub, False)
        

    cdef _setBound(self, varidx, b, bint lower_bound):
        
        if b is None:
            b = neg_arinfty if lower_bound else pos_arinfty

        elif type(b) is list:
            b = [(-infty if lower_bound else infty) 
                 if be is None else be for be in b]

        elif type(b) is ndarray:
            b[~isfinite(b)] = -infty if lower_bound else infty


        if lower_bound:
            self._stackOnInterpretedKeyValuePair(
                self._lower_bound_keys, self._lower_bound_values, varidx, b, &self._lower_bound_count)
        else:
            self._stackOnInterpretedKeyValuePair(
                self._upper_bound_keys, self._upper_bound_values, varidx, b, &self._upper_bound_count)
            

    cdef applyVariableBounds(self):
        
        assert self.lp != NULL

        cdef ar[int, mode="c"] I
        cdef ar[double, mode="c"] V
               
        # First the lower bounds; thus we can use set_unbounded on them
        I, V = self._getArrayPairFromKeyValuePair(
            self._lower_bound_keys, self._lower_bound_values, self._lower_bound_count)

        for 0 <= i < I.shape[0]:
            if V[i] == -infty:
                set_unbounded(self.lp, I[i] + 1)
            else:
                set_lowbo(self.lp, I[i] + 1, V[i])

        # Now set the upper bounds; this is trickier as set_unbounded
        # undoes the lower bound

        I, V = self._getArrayPairFromKeyValuePair(
            self._upper_bound_keys, self._upper_bound_values, self._upper_bound_count)

        cdef double lp_infty = get_infinite(self.lp)
        cdef double lb

        for 0 <= i < I.shape[0]:
            if V[i] == infty:
                lb = get_lowbo(self.lp, I[i] + 1)
                set_unbounded(self.lp, I[i] + 1)
            
                if lb != lp_infty:
                    set_lowbo(self.lp,  I[i] + 1, lb)
            else:
                set_upbo(self.lp, I[i] + 1, V[i])


    ############################################################
    # Helper functions for turning dictionaries or tuple-lists into an
    # index array + value array.

    cdef ar _attemptArrayList(self, list l):

        # Now we need to specify that only lists of lists or lists of
        # arrays are allowed, not lists of tuples.  
    
        cdef bint is_l, is_a
        cdef int all_size = -1, size
        cdef bint inconsistent_lengths = False
        cdef ar A

        for t in l:
            
            is_l = isinstance(t, list)
            is_a = isinstance(t, ndarray)
            
            if not (is_l or is_a):
                return None

            elif is_l:
                if not isnumericlist(<list>t):
                    return None

                size = len(<list>t)

            elif is_a:
                A = <ar>t
                
                if A.ndim != 1:
                    return None
                
                size = A.shape[0]

            else:
                return None
            
            # Keep track of the sizes
            if all_size == -1:
                all_size = size
            else:
                if all_size != size:
                    inconsistent_lengths = True

        if inconsistent_lengths:
            raise ValueError("Lengths of sublist in list of lists/arrays not consistent.")
        else:
            return array(l, npfloat)
                    
    cdef tuple _getArrayPairFromDict(self, dict d):
        # Builds an array pair from a dictionary

        cdef list key_buf = [], val_buf = []
        cdef size_t idx_count = 0

        for k, v in d.iteritems():
            self._stackOnInterpretedKeyValuePair(key_buf, val_buf, k, v, &idx_count)

        cdef ar I = empty(idx_count, npint)
        cdef ar V = empty(idx_count, npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)

        return (I, V)

    cdef tuple _getArrayPairFromTupleList(self, list l):
        # Builds an array pair from a tuple

        cdef list key_buf = [], val_buf = []
        cdef size_t idx_count = 0

        for k, v in l:
            self._stackOnInterpretedKeyValuePair(key_buf, val_buf, k, v, &idx_count)

        cdef ar I = empty(idx_count, npint)
        cdef ar V = empty(idx_count, npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)

        return (I, V)

    cdef tuple _getArrayPairFromKeyValuePair(self, list key_buf, list val_buf, size_t idx_count):
        cdef ar I = empty(idx_count, npint)
        cdef ar V = empty(idx_count, npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)
        
        return (I, V)
    
    cdef _stackOnInterpretedKeyValuePair(self, list key_buf, list val_buf, k, v, size_t *count):
        # Appends interpreted key value pairs to the lists

        cdef ar tI
        cdef ar[double] tV
        cdef size_t i

        cdef ar[size_t, ndim=2, mode="c"] B

        if isposint(k):
            tV = self._resolveValues(v, True)
            
            if tV.shape[0] != 1:
                raise ValueError("Scalar index must specify only one value.")

            key_buf.append(k)
            val_buf.append(tV[0])
            count[0] += 1

        elif (type(k) is str or type(k) is tuple 
              or type(k) is list or type(k) is ndarray):

            tV = self._resolveValues(v, True)
            tI = self._resolveIdxBlock(k, tV.shape[0])
            
            key_buf.append(tI)
            val_buf.append(tV)

            assert not tI is None

            if tI.ndim == 1:
                if tI.dtype is not npint:
                    ptI = tI
                    tI = array(tI, npint)
                    if (tI != ptI).any():
                        raise ValueError("Could not convert index array into array of integers.")
                    
                count[0] += tI.shape[0]
            elif tI.ndim == 2:
                count[0] += tI[0,1] - tI[0,0]

        elif k is None:
            tV = self._resolveValues(v, True)
            tI = self._resolveIdxBlock(None, tV.shape[0])
            
            B = empty(range_tuple_size, npuint)
            B[0,0] = 0
            B[0,1] = self.n_columns

            key_buf.append(B)
            val_buf.append(tV)

            count[0] += self.n_columns

        else:
            raise TypeError("Error interpreting key/value pair as index/value pair.")

        assert len(key_buf) == len(val_buf), "k:%d != v:%d" % (len(key_buf), len(val_buf))


    cdef _fillIndexValueArraysFromIntepretedStack(self, ar I_o, ar V_o, list key_buf, list val_buf):
        
        cdef ar[int, mode="c"] I = I_o
        cdef ar[double, mode="c"] V = V_o

        cdef ar[int] tI
        cdef ar[double] tV
        
        cdef ar[size_t, ndim=2, mode="c"] B

        cdef size_t i, j, d

        assert len(key_buf) == len(val_buf), "k:%d != v:%d" % (len(key_buf), len(val_buf))

        cdef size_t ii = 0

        for 0 <= i < len(key_buf):
            k = key_buf[i]
            v = val_buf[i]

            if isnumeric(k):
                I[ii] = <size_t>k
                V[ii] = <double>v
                ii += 1

            elif type(k) is ndarray:
                
                d = (<ar>k).ndim
                
                if d == 1:

                    tI = <ar>k
                    tV = <ar>v

                    if tI.shape[0] != 1 and tV.shape[0] == 1:
                        for 0 <= j < tI.shape[0]:
                            I[ii] = tI[j]
                            V[ii] = tV[0]
                            ii += 1

                    elif tI.shape[0] == tV.shape[0]:
                        for 0 <= j < tI.shape[0]:
                            I[ii] = tI[j]
                            V[ii] = tV[j]
                            ii += 1
                    else:
                        assert False

                elif d == 2:
                    B = <ar>k
                    tV = <ar>v
                    
                    if tV.shape[0] == 1:
                        for 0 <= j < B[0,1] - B[0,0]:
                            I[ii] = B[0,0] + j
                            V[ii] = tV[0]
                            ii += 1
                    elif tV.shape[0] == (B[0,1] - B[0,0]):
                        for 0 <= j < tV.shape[0]:
                            I[ii] = B[0,0] + j
                            V[ii] = tV[j]
                            ii += 1
                    else:
                        assert False

                else:
                    assert False
            else:
                assert False
            
        assert ii == I.shape[0], "ii=%d, I.shape[0]=%d" % (ii, I.shape[0])



    ############################################################
    # Now stuff for solving the model

    cdef setupLP(self, dict option_dict):
        
        ########################################
        # Go through and configure things depending on the options

        ####################
        # Presolve
        cdef unsigned long n, presolve = 0

        for k, n in presolve_flags.iteritems():
            if option_dict[k]:
                presolve += n

        ####################
        # Pricing

        pricer = option_dict["pricer"].lower()

        if type(pricer) is not str or pricer not in pricer_lookup:
            raise ValueError("pricer option must be one of %s."
                             % (','.join(["'%s'" % p for p in pricer_lookup.iterkeys()])))

        cdef int pricer_option = pricer_lookup[pricer]

        # Now see if there are other flags in this mix
        for k, n in pricer_flags.iteritems():
            if option_dict[k]:
                pricer_option += n


        ####################
        # Scaling

        scaling = option_dict["scaling"].lower()

        if type(scaling) is not str or scaling not in scaling_lookup:
            raise ValueError("scaling option must be one of %s."
                             % (','.join(["'%s'" % s for s in scaling_lookup.iterkeys()])))

        cdef int scaling_option = scaling_lookup[scaling]

        # Now see if there are other flags in this mix
        for k, n in scaling_flags.iteritems():
            if option_dict[k]:
                scaling_option += n

        ####################
        # Verbosity

        if option_dict["verbosity"] not in [1,2,3,4,5]:
            raise ValueError("Verbosity level must be 1,2,3,4, or 5 (highest).")

        # Options are vetted now; basis and others might not be 

        ########################################
        # Now set up the LP

        if self.lp == NULL:
            self.lp = make_lp(self.n_rows, self.n_columns)

            if self.lp == NULL:
                raise MemoryError("Out of memory creating internal LP structure.")
        else:
            if not resize_lp(self.lp, self.n_rows, self.n_columns):
                raise MemoryError("Out of memory resizing internal LP structure.")
            
        ####################
        # Constraints

        self.applyAllConstraints()

        ####################
        # Variable Bounds

        self.applyVariableBounds()

        ####################
        # Objective

        self.applyObjective()

        ####################
        # Set the basis/guess

        # Stuff for the basis setting
        cdef ar b, g

        cdef size_t basic_basis_size = 1 + self.n_columns
        cdef size_t full_basis_size = basic_basis_size + self.n_rows

        cdef ar[int, mode="c"]    start_basis = None
        cdef ar[double, mode="c"] guess_vect  = None

        # possibly set the start basis
        if "basis" in option_dict:
            if presolve != 0:
                warnings.warn("Presolve must not be active when combined with setting basis; ignoring presolve.")
                presolve = 0

            basis = option_dict["basis"]

            if type(basis) is ndarray:
                b = basis
            elif type(basis) is list:
                b = array(basis)
            else:
                raise TypeError("Basis must be either ndarray or list.")

            if b.ndim != 1 or b.shape[0] not in [full_basis_size, basic_basis_size]:
                raise ValueError("Basis must be 1d array of length 1 + num_columns or 1 + num_columns [+ num_rows]")

            if b.dtype != npint:
                warnings.warn("Basis not an integer array/list, converting.")
                start_basis = npint(b)
            else:
                start_basis = b.ravel()

        elif "guess" in option_dict:
            if presolve != 0:
                warnings.warn("Presolve must not be active when combined with setting basis; ignoring presolve.")
                presolve = 0

            start_basis = self._getBasisFromGuess(option_dict["guess"], option_dict["error_on_bad_guess"])

        ########################################
        # Set all the options
            
        set_presolve(self.lp, presolve, 100)
        set_pivoting(self.lp, pricer_option)
        set_scaling(self.lp, scaling_option)
        set_verbose(self.lp, option_dict["verbosity"])

        if start_basis is not None:
            assert start_basis.shape[0] in [full_basis_size, basic_basis_size]
            set_basis(self.lp, <int*>start_basis.data, start_basis.shape[0] == full_basis_size)

        ####################
        # Clear out all the temporary stuff 
        self._clear(True)


    cdef ar _getBasisFromGuess(self, guess, bint error_on_bad_guess):
        # Note: it is not safe to run this function if the presolve
        # has not been set up

        cdef ar[double, mode="c"] guess_vect
        cdef ar[int, mode="c"] start_basis

        if type(guess) is ndarray or (type(guess) is list and isnumericlist(guess)):
            # We have a full and complete guess

            g = asarray(guess, dtype=npfloat)

            if g.ndim != 1 or g.shape[0] != self.n_columns:
                raise ValueError("Guess must be 1d array of length num_columns.")

            guess_vect = empty(1 + self.n_columns, npfloat)
            guess_vect[0] = 0
            guess_vect[1:] = g

            # Now try to create the basis
            start_basis = zeros(1 + self.n_columns + self.n_rows, npint)

            if not guess_basis(self.lp, <double*>guess_vect.data, <int*>start_basis.data):

                error_msg = "Finding starting basis from guess vector failed; discarding."

                if error_on_bad_guess:
                    raise LPSolveException(error_msg)
                else:
                    warnings.warn(error_msg)
                    return None
            
            return start_basis

        # Made it here, so the basis must be a different type
        
        cdef ar[int, mode="c"]    I
        cdef ar[double, mode="c"] V
        cdef tuple t
        
        cdef size_t prev_n_cols = self.n_columns

        if type(guess) is dict:
            I, V = self._getArrayPairFromDict(guess)
        elif type(guess) is list and istuplelist(guess):
            I, V = self._getArrayPairFromTupleList(guess)
        elif type(guess) is tuple and len(<tuple>guess) == 2:
            I, V = self._getArrayPairFromTupleList([guess])
        elif isnumeric(guess):
            I = array([0], npint)
            V = array([guess], npfloat)
        else:
            raise ValueError("Type of guess not recognized.")

        # Make sure the number of columns hasn't changed
        if prev_n_cols != self.n_columns:
            self.n_columns = prev_n_cols
            raise ValueError("Index in guess out of bounds.")

        # Now see if it's a full guess and we can punt it back
        if isfullindexarray(I, self.n_columns):
            return self._getBasisFromGuess(V[argsort(I)], error_on_bad_guess)
        else:
            raise ValueError("Variable guess not complete.")
        
            # We could copy the LP, then set the parameters given via
            # identical lower/upper bounds, then allow the presolve to
            # modify the model as needed.  But that's for later.

    def solve(self, **options):
        """
        Solves the given model.  `mode` may be either "minimize"
        (default) or "maximize".

        Configuration
        ========================================
        
        Any of the options available to `setOption()` may also be
        passed in as keyword arguments.  These affect only the current
        run (as opposed to `setOption()`, which affects all subsequent
        runs of `solve()`).

        Additional keyword options, unique to `solve()`, are::

          basis:
            Initialize the LP with the given basis.  This must be an
            array of integers of size ``1 + num_columns`` or ``1 +
            num_columns + num_rows`` from a previous run of the LP.
            Use `getBasis()` to retrieve the basis after a given run.
            This can speed up the time to find a solution
            substantially.

          guess:
            Initialize the LP with guessed starting values of the
            variables.  This must be an array of length
            ``num_columns`` and should be a feasible solution given
            the constraints.

            The LP is initialized by first constructing a basis given
            the guessed variable values, then initializing it with
            that basis.  If both basis and guess are given as options,
            guess is ignored. 

            Note that if the parameter 'error_on_bad_guess' is True,
            an exception is raised if the guess fails.  The default is
            False, which causes a warning to be generated.
        
        """

        ########################################
        # Get the current options dict

        cdef dict option_dict = self.getOptionDict()

        # Make sure that the options given are all valid
        cdef set okay_options = set(option_dict.iterkeys())
        okay_options |= set(["basis", "guess"])

        cdef str k, k1

        for k, v in options.iteritems():
            kl = k.lower()

            if kl not in okay_options:
                raise ValueError("Option '%s' not recognized." % k)

            option_dict[kl] = v

        
        ########################################
        # Set up the LP
        
        self.setupLP(option_dict)

        #self.print_lp()

        cdef int ret = solve(self.lp)
        
        ########################################
        # Check the error codes

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

        # And we're done
        


    def getSolution(self, indices = None):
        """
        Returns the final values of the variables in the constraints.

        The following are valid values of `indices`:

          None (default):
            Retrieves the full array of final variables.

          Variable block name:
            Retrieves an array of the values in the given block.

          List or array of indices:
            Retrieves the values corresponding to the given indices.

          2-tuple:
            The 2-tuple must specify the (low, high) bounds of a
            variable block; this block is returned.

          Scalar index:
            Returns a single value corresponding to the specified index.
        """

        if self.lp == NULL:
            raise LPSolveException("Final variables available only after solve() is called.")

        # Okay, now we've got it

        cdef double *vars

        get_ptr_variables(self.lp, &vars)

        cdef tuple t
        cdef ar[size_t, ndim=2, mode="c"] idx_bounds
        cdef ar[double, mode="c"] res
        cdef ar[int] idx_request
        cdef list l
        cdef int idx, idx_lb, idx_ub
        cdef size_t i

        if indices is None:
            res = empty(self.n_columns, npfloat)
            for 0 <= i < res.shape[0]:
                res[i] = vars[i]
            return res

        elif type(indices) is str:

            try:
                idx_bounds = self._named_index_blocks[indices]
            except KeyError:
                raise ValueError("Variable block '%s' not defined." % indices)
            
            res = empty(idx_bounds[0,1] - idx_bounds[0,0], npfloat)
    
            for 0 <= i < res.shape[0]:
                res[i] = vars[idx_bounds[0,0] + i]

            return res

        elif type(indices) is list:
            l = <list>indices
            if not isposintlist(l):
                raise ValueError("Requested index list must contain only valid indices.")
            
            res = empty(len(l), npfloat)

            for 0 <= i < res.shape[0]:
                idx = l[i]
                
                if idx < 0 or idx >= self.n_columns:
                    raise ValueError("Variable index not valid: %d" % idx)

                res[i] = vars[idx]

            return res

        elif type(indices) is ndarray:
            idx_request = asarray(indices, npint)
            
            res = empty(idx_request.shape[0], npfloat)
            
            for 0 <= i < res.shape[0]:
                idx = idx_request[i]

                if idx < 0 or idx >= self.n_columns:
                    raise ValueError("Variable index not valid: %d" % idx)

                res[i] = vars[idx]
            
            return res

        elif type(indices) is tuple:
            t = indices
            self._validateIndexTuple(t)

            idx_lb = t[0]
            idx_ub = t[1]

            res = empty(idx_ub - idx_lb, npfloat)
    
            for 0 <= i < res.shape[0]:
                res[i] = vars[idx_lb + i]

            return res

        elif isposint(indices):
            idx = indices

            if idx < 0 or idx >= self.n_columns:
                raise ValueError("Variable index not valid: %d" % idx)
            
            return vars[idx]

        else:
            raise ValueError("Type of `indices` argument not recognized.")

    def getSolutionDict(self):
        """
        Returns a dictionary of all the solutions to all of the
        previously named variable blocks.
        """

        cdef dict ret = {}

        for k in self._named_index_blocks.iterkeys():
            ret[k] = self.getSolution(k)
            
        return ret


    cpdef real getObjectiveValue(self):
        """
        Returns the value of the objective function of the LP.
        """

        if self.lp == NULL:
            raise LPSolveException("Final variables available only after solve() is called.")

        return get_objective(self.lp)

    def getBasis(self, include_dual_basis = True):
        """
        Returns the basis from the previous run.
        """

        if self.lp == NULL:
            raise LPSolveException("Info available only after solve() is called.")

        cdef ar[int, mode="c"] basis = empty(
            1 + self.n_columns + (self.n_rows if include_dual_basis else 0), npint)

        if not get_basis(self.lp, <int*>basis.data, include_dual_basis):
            raise LPSolveException("Unknown error while retrieving basis.")

        return basis

    cpdef print_lp(self):

        self.setupLP(self.getOptions())
        print_lp(self.lp)
        

    def getInfo(self, info):
        """
        Returns a specific statistic from the latest run of the lp.
        The available statistics are 
        """
        
        if self.lp == NULL:
            raise LPSolveException("Info available only after solve() is called.")

        cdef int ret_stat

        try:
            ret_stat = info_lookup[info.lower()]
        except KeyError:
            raise ValueError("info must be one of: %s" % 
                             ", ".join(info_lookup.iterkeys()))

        if ret_stat == nIterations:
            return get_total_iter(self.lp)
        else:
            assert False
            

    ############################################################
    # Methods for dealing with the constraint buffers

    cdef setConstraint(self, size_t row_idx, ar idx, ar row, int ctype, rhs):
        
        # First get the right cstr
        cdef _Constraint* cstr = self.getConstraintStruct(row_idx)

        if cstr == NULL:
            raise MemoryError

        setupConstraint(cstr, row_idx, idx, row, ctype, rhs)

    cdef _addConstraint(self, ar idx, ar row, int ctype, rhs):
        cdef size_t row_idx = self.n_rows
        self.setConstraint(row_idx, idx, row, ctype, rhs)
        return row_idx

    cdef _Constraint* getConstraintStruct(self, size_t row_idx):
        
        # First see if our double-buffer thing is ready to go
        cdef size_t buf_idx = <size_t>(row_idx // cStructBufferSize)
        cdef size_t idx     = <size_t>(row_idx % cStructBufferSize)
        cdef size_t new_size

        cdef _Constraint* buf
        cdef size_t i

        # Ensure proper sizing of constraint buffer
        if buf_idx >= self.current_c_buffer_size:
            
            # Be agressive, since we anticipate growing incrementally,
            # and each one is a lot of constraints

            new_size = 128 + 2*buf_idx
            new_size -= (new_size % 128)

            assert new_size > 0
            assert new_size > buf_idx

            if self._c_buffer == NULL:
                self._c_buffer = <_Constraint**>malloc(new_size*sizeof(_Constraint*))
            else:
                self._c_buffer = <_Constraint**>realloc(self._c_buffer, new_size*sizeof(_Constraint*))
            
            if self._c_buffer == NULL:
                return NULL  

            for self.current_c_buffer_size <= i < new_size:
                self._c_buffer[i] = NULL

            #memset(&self._c_buffer[self.current_c_buffer_size], 0,
            #        (new_size - self.current_c_buffer_size)*sizeof(_Constraint*))

            self.current_c_buffer_size = new_size

        # Now make sure that the buffer is ready
        buf = self._c_buffer[buf_idx]
        
        if buf == NULL:
            buf = self._c_buffer[buf_idx] = \
                <_Constraint*>malloc(cStructBufferSize*sizeof(_Constraint))
            
            if buf == NULL:
                raise MemoryError

            memset(buf, 0, cStructBufferSize*sizeof(_Constraint))

        # Now finally this determines the new model size
        if row_idx >= self.n_rows:
            self.n_rows = row_idx + 1
        
        return &buf[idx]

    cdef applyAllConstraints(self):
        assert self.lp != NULL

        cdef size_t i, j
        cdef _Constraint *buf

        # turn on row adding mode
        set_add_rowmode(self.lp, True)

        # This enables the slicing to work properly
        cdef int* countrange = <int*>malloc((self.n_columns+1)*sizeof(int))

        for 0 <= i <= self.n_columns:
            countrange[i] = i

        for 0 <= i < self.current_c_buffer_size:
            buf = self._c_buffer[i]
        
            if buf == NULL:
                continue

            for 0 <= j < cStructBufferSize:
                if inUse(&buf[j]):
                    # debug note; not in here
                    setInLP(&buf[j], self.lp, self.n_columns, countrange)

        free(countrange)

        # Turn off row adding mode
        set_add_rowmode(self.lp, False)

    cdef void clearConstraintBuffers(self):

        cdef size_t i, j
        cdef _Constraint* buf

        if self._c_buffer == NULL:
            return

        for 0 <= i < self.current_c_buffer_size:

            buf = self._c_buffer[i]
            
            if buf == NULL:
                continue

            for 0 <= j < cStructBufferSize:
                if inUse(&buf[j]):
                    clearConstraint(&buf[j])

            free(buf)

        free(self._c_buffer)
        
        self.current_c_buffer_size = 0
        self._c_buffer = NULL


################################################################################
# Stuff to handle the constraints
################################################################################

# Defines a constraint struct that allows for buffered adding of
# constraints.  This shields the user from many of the lower-level
# considerations with lp_solve


######################################################################
# Now functions and a struct dealing with the constraints

cdef struct _Constraint:
    # Dealing with the indices; if index_range_mode is true, then the
    # indices refer to a range rather than individual indices
    int *indices
    int index_range_start  # the first index of the length n block
                           # that is the indices. Negative -> not
                           # used.

    # The values
    double *values

    # The total size
    size_t n
    int ctype
    double rhs1, rhs2
    size_t row_idx

########################################
# Methods to deal with this constraint

cdef inline setupConstraint(_Constraint* cstr, size_t row_idx, ar idx, ar row, int ctype, rhs):

    # see if we need to clear things
    assert cstr.n == 0
    assert cstr.indices == NULL
    assert cstr.values == NULL

    ########################################
    # Check possible bad configurations of ctype, rhs
    cstr.ctype = ctype

    if cstr.ctype in [constraint_leq, constraint_geq, constraint_equal]:
        if not isnumeric(rhs):
            raise TypeError("Constraint type '%s' requires right hand side to be scalar." % getReverseCType(ctype))

        cstr.rhs1 = <double>rhs
    elif cstr.ctype == constraint_in:
        if type(rhs) is tuple and len(<tuple>rhs) == 2:
            cstr.rhs1, cstr.rhs2 = (<tuple>rhs)
        elif type(rhs) is list and len(<list>rhs) == 2:
            cstr.rhs1, cstr.rhs2 = (<list>rhs)
        else:
            raise TypeError("Constraint type '%s' requires right hand side to be either 2-tuple or 2-list."  % getReverseCType(ctype))
    else:
        assert False

    ############################################################
    # Set the row indices
    cstr.row_idx = row_idx
    
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

            if row.shape[0] == 1:
                fill_mode = True
                fill_value = row[0]
            else:
                assert row.shape[0] == cstr.n, "r: %d != %d n" % (row.shape[0], cstr.n)
        else:
            assert False

    else:
        cstr.n = row.shape[0]

    # Set the indices
    if idx is None or idx_range_mode:
        cstr.indices = NULL
    else:
        cstr.indices = <int*> malloc((cstr.n+1)*sizeof(int))

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
    cdef ar[double] a = a_o

    assert cstr.values != NULL
    assert cstr.n >= a.shape[0]

    for 0 <= i < a.shape[0]:
        cstr.values[i+1] = a[i]


cdef inline void fillValues(_Constraint* cstr, double v):

    cdef size_t i

    for 1 <= i <= cstr.n:
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
            if (ua != a).any():
                raise ValueError("Error converting index array to 32bit integers.")

            copyIntoIndices_uint32(cstr, ua)
        else:
            ua = uint64(a)
            if not (ua != a).any():
                raise ValueError("Error converting index array to 64bit integers.")

            copyIntoIndices_uint64(cstr, ua)
            
cdef inline copyIntoIndices_int32(_Constraint *cstr, ar a_o):

    cdef ar[int32_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.indices[i+1] = a[i] + 1

cdef inline copyIntoIndices_uint32(_Constraint *cstr, ar a_o):

    cdef ar[uint32_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.indices[i+1] = a[i] + 1

cdef inline copyIntoIndices_int64(_Constraint *cstr, ar a_o):

    cdef ar[int64_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.indices[i+1] = a[i] + 1

cdef inline copyIntoIndices_uint64(_Constraint *cstr, ar a_o):

    cdef ar[uint64_t] a = a_o
    cdef size_t i

    with cython.boundscheck(False):
        for 0 <= i < a.shape[0]:
            cstr.indices[i+1] = a[i] + 1

############################################################
# Now the routines to add this to the model

cdef inline setInLP(_Constraint *cstr, lprec* lp, size_t n_cols, int *countrange):

    if cstr.n == 0:
        return

    cdef size_t i

    # Ensure that the columns and all are sized up correctly
    if cstr.indices == NULL and cstr.index_range_start == -1:
        cstr.index_range_start = 1

    # Vanila constraint
    if cstr.ctype in [constraint_leq, constraint_geq, constraint_equal]:
        _setRow(cstr, lp, cstr.ctype, cstr.rhs1, countrange)

    # range constraint
    elif cstr.ctype == constraint_in:

        if cstr.rhs1 < cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs2, countrange)
            set_rh_range(lp,  cstr.row_idx+1, cstr.rhs2 - cstr.rhs1)

        elif cstr.rhs1 > cstr.rhs2:
            _setRow(cstr, lp, constraint_leq, cstr.rhs1, countrange)
            set_rh_range(lp,  cstr.row_idx+1, cstr.rhs1 - cstr.rhs2)

        else:
            _setRow(cstr, lp, constraint_equal, cstr.rhs1, countrange)
    else:
        assert False  # no silent fail


cdef inline _setRow(_Constraint *cstr, lprec *lp, int ctype, double rhs, int *countrange):

    cdef size_t i

    # Need to accommidate the start-at-1 indexing
    if cstr.indices == NULL:
        if cstr.index_range_start != -1:

            set_rowex(lp, cstr.row_idx+1, cstr.n, cstr.values+1, countrange + cstr.index_range_start)
        else:
            set_row(lp, cstr.row_idx+1, cstr.values)
    else:
        set_rowex(lp, cstr.row_idx+1, cstr.n, cstr.values+1, cstr.indices+1)

    set_constr_type(lp, cstr.row_idx+1, ctype)
    set_rh(lp, cstr.row_idx+1, rhs)


cdef inline void clearConstraint(_Constraint *cstr):
    if cstr.indices != NULL: free(cstr.indices)
    if cstr.values  != NULL: free(cstr.values)
    
    cstr.indices = NULL
    cstr.values = NULL
    cstr.n = 0

cdef inline bint inUse(_Constraint *cstr):
    return (cstr.n != 0)
