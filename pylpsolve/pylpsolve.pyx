from numpy cimport ndarray as ar, uint_t, float_t

from numpy import empty, ones, zeros, uint, arange, isscalar, amax, float as npfloat

from constraintstruct cimport _Constraint, setupConstraint, \
    setInLP, clearConstraint, isInUse

import warnings

cimport cython

ctypedef double real
ctypedef unsigned char ecode

import optionlookup

############################################################
# An important check

assert sizeof(uint_t) == sizeof(int)
assert sizeof(float_t) == sizeof(double)

######################################################################
# A few early binding things

cdef dict default_options = optionlookup._default_options
cdef dict presolve_flags  = optionlookup._presolve_flags
cdef dict pricer_lookup   = optionlookup._pricer_lookup
cdef dict pricer_flags    = optionlookup._pricer_flags

cdef double infty = 1e30

######################################################################
# LPSolve constants

cdef extern from "lpsolve/lp_lib.h":
    ctypedef void lprec

    cdef:
        lprec* make_lp(int rows, int cols)
        void delete_lp(lprec*)

        ecode resize_lp(lprec *lp, int rows, int columns)

        ecode set_obj_fn(lprec*, real* row)
        ecode set_obj_fnex(lprec*, int count, real*row, int *colno)

        ecode add_constraint(lprec*, real* row, int ctype, real rh)
        ecode add_constraintex(lprec*, int count, real* row, int *colno, int ctype, real rh)

        ecode set_rowex(lprec *lp, int row_no, int count, real *row, int *colno)
        ecode set_constr_type(lprec *lp, int row, int con_type)
        ecode set_rh(lprec *lp, int row, real value)
        ecode set_rh_range(lprec *lp, int row, real deltavalue)

        ecode set_add_rowmode(lprec*, unsigned char turn_on)
        
        ecode set_lowbo(lprec*, int column, real value)
        ecode set_upbo(lprec*, int column, real value)
        real get_lowbo(lprec*, int column)
        real get_upbo(lprec*, int column)
        real get_infinite(lprec*)
        ecode set_bounds(lprec*, int column, real lower, real upper)
        ecode set_unbounded(lprec *lp, int column)

        void set_presolve(lprec*, int do_presolve, int maxloops)

        int get_Ncolumns(lprec*)

        ecode get_variables(lprec*, real *var)
        real get_objective(lprec*)

        void set_pivoting(lprec*, int rule)

        int solve(lprec *lp)
        
        int print_lp(lprec *lp)


############################################################
# Structs for buffering the constraints

cdef extern from "Python.h":
    void* malloc "PyMem_Malloc"(size_t n)
    void* realloc "PyMem_Realloc"(void *p, size_t n)
    void free "PyMem_Free"(void *p)


cdef extern from "stdlib.h":
    void memset(void *p, char value, size_t n)


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

    cdef list _upper_bound_keys, _upper_bound_valuse
    cdef size_t _upper_bound_count
    
    # The objective function
    cdef list _obj_func_keys
    cdef list _obj_func_values
    cdef bint _obj_func_specified
    cdef size_t _obj_func_n_vals_count 

    # Methods relating to named variable group
    cdef dict _named_index_blocks
    cdef bint _nonindexed_blocks_present, _user_warned_about_nonindexed_blocks

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

    cdef _clear(self, bint restartable_mode):
        # If restartable mode, only delete all the temporary stuff
        # but still allow the user to tweak the model and resume the LP

        self._lower_bound_keys   = []
        self._lower_bound_values = []
        self._lower_bound_count  = 0

        self._upper_bound_keys   = []
        self._upper_bound_valuse = []
        self._upper_bound_count  = 0

        self._obj_func_keys   = []
        self._obj_func_values = []
        self._obj_func_specified = False
        self._obj_func_n_vals_count = 0

        self.clearConstraintBuffers()

        if not restartable_mode:  
            if self.lp != NULL:
                delete_lp(self.lp)
                self.lp = NULL

            self.n_rows = 0
            self.n_columns = 0
            
            self._named_index_blocks = {}
            self._user_warned_about_nonindexed_blocks = False
            self._nonindexed_blocks_present = False


    ############################################################
    # Methods concerned with getting and setting the options

    def getOptions(self):
        return self.options.copy()

    def setOption(self, str name, value):
        """
        Presolve options
        ==================================================
        
        Available presolve options are::

          presolve_none:
            No presolve at all.

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
        
        Available pricer options::
    
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
        """

        cdef str n = name.lower()
        
        if n not in self.options:
            raise ValueError("Option '%s' not valid." % name)

        self.options[n] = value


    ############################################################
    # Methods relating to variable indices

    def getVariableIndexBlock(self, size_t size, str name = None):
        """
        Returns a new or current block of 'size' variable indices to
        be used in the current LP.
        
        If `name` is not None, then other functions
        (e.g. addConstraint) can accept this accept this string as the
        index argument.

        # Put in use cases

        """
        
        self._getVariableIndexBlock(size, name)
    

    cdef ar _getVariableIndexBlock(self, size_t size, str name):
        cdef ar idx

        if name is None:
            idx = uint(arange(self.n_columns, self.n_columns + size))
            self.checkColumnCount(self.n_columns + size, True)
            return idx
        else:
            if name in self._named_index_blocks:
                idx = self._named_index_blocks[name]
                if idx.shape[0] != size:
                    raise ValueError("Requested size (%d) does not match previous size (%d) of index block '%s'"
                                     % (size, idx.shape[0], name))
                return idx
            else:
                idx = uint(arange(self.n_columns, self.n_columns + size))
                self.checkColumnCount(self.n_columns + size, True)
                self._named_index_blocks[name] = idx
                return idx


    def getVariableIndex(self, str name = None):
        """
        Returns a single unused index (unless name refers to a
        previously named index).
        """

        if name is None:
            ret_val = self.n_columns
            self.checkColumnCount(self.n_columns + 1)
            return ret_val
        else:
            return self.getVariableIndexBlock(1, name)[0]


    ########################################
    # For internally resolving things
    cdef ar _resolveIdxBlock(self, idx, size_t n):
    
        if idx is None:
            self.checkColumnCount(n, True)
            return None

        cdef ar ar_idx
        cdef long ar_idx

        if type(idx) is str:
            return self.getVariableIndexBlock(<str>idx, n)

        elif type(idx) is ndarray:
            ar_idx = idx

            if ar_idx.ndim != 1:
                raise ValueError("Index array must be 1d vector.")
            if ar_idx.shape[0] not in (n, 1):
                raise ValueError("Length of index array (%d) must equal length of values (%d) or 1." 
                                 % (ar_idx.shape[0], n))

            self.checkColumnCount(amax(ar_idx), False)
            return ar_idx.copy()  # Need to own the data

        elif type(idx) is list or type(idx) is tuple:
            try:
                ar_idx = array(idx)
            except Exception, e:
                raise ValueError("Error converting index list to 1d array: %s" % str(e))
            
            if ar_idx.ndim != 1:
                raise ValueError("Error interpreting index list: Not 1 dimensional.")
            if ar_idx.shape[0] not in (n, 1):
                raise ValueError("Length of index list (%d) must equal length of values (%d) or 1." 
                                 % (ar_idx.shape[0], n))
            
            self.checkColumnCount(amax(ar_idx), False)
            return ar_idx

        elif isnumeric(idx):
            v_idx = <size_t>idx
            if v_idx != idx:
                raise ValueError("%s not valid as nonnegative index. " % str(idx))

            self.checkColumnCount(v_idx)
            ar_idx = array([idx], dtype=uint)
            return ar_idx

        else:
            raise TypeError("Type of index (%s) not recognized; must be scalar, list, tuple, str, or array." % type(idx))


    cdef ar _resolveValues(self, v, bint ensure_1d):
        cdef ar ret

        if type(v) is ndarray:
            ret = npfloat(v)
        elif type(v) is list or type(v) is tuple:
            ret = array(v, dtype=npfloat)
        elif isnumeric(v):
            ret = array([v],dtype=npfloat)
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

        if indexed:
            self._nonindexed_blocks_present = True


    cdef _warnAboutNonindexedBlocks(self):
        if not self._user_warned_about_nonindexed_blocks:
            warnings.warn("Non-indexed variable block present which does not span columns; "
                          "setting to lowest-indexed columns.")
            self._user_warned_about_nonindexed_blocks = True
            


    ############################################################
    # Methods dealing with constraints

    def addConstraint(self, coefficients, str ctype, rhs):
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

        # What we do depends on the type
        coefftype = type(coefficients)

        if coefftype is tuple:
            t_coeff = <tuple>coefficients

            # Make sure it's split into index, value pair
            if len(t_coeff) != 2:
                raise TypeError("coefficients should be either a single array, "
                                "a dictionary, or a 2-tuple with (index block, values)")

            return self._addConstraintArray(t_coeff[0], t_coeff[1], ctype, rhs)

        elif coefftype is ndarray:
            return self._addConstraintArray(None, coefficients, ctype, rhs)

        elif coefftype is list:
            # Two possible ways to interpret a list; as a sequence of
            # tuples or as a representation of an array; if it is a
            # sequence of 2-tuples, intepret it this way 
            

            # Test and see if it's a list of sequences or numerical list
            is_list_sequence = self._isTupleList(<list>coefficients)

            if is_list_sequence: 
                is_numerical_sequence = False
            else:
                is_numerical_sequence = self._isNumericalList(<list>coefficients)                    
                
            if is_list_sequence:
                return self._addConstraintTupleList(<list>coefficients, ctype, rhs)
            elif is_numerical_sequence:
                return self._addConstraintArray(None, coefficients, ctype, rhs)
            else:
                raise TypeError("Coefficient list must be either list of scalars or list of 2-tuples.")

        elif coefftype is dict:
            return self._addConstraintDict(<dict>coefficients, ctype, rhs)

        else:
            raise TypeError("Type of coefficients not recognized; must be dict, list, 2-tuple, or array.")


    cdef _addConstraintArray(self, t_idx, t_val, str ctype, rhs):
        # If the values can be interpreted as an array

        cdef ar A = self._resolveValues(t_val, False)
        cdef size_t i
        cdef ar[double] rhs_a

        if A.ndim == 1:
            idx = self._resolveIdxBlock(t_idx, A.shape[0])
            return self._addConstraint(idx, A, ctype, rhs)

        elif A.ndim == 2:
            idx = self._resolveIdxBlock(t_idx, A.shape[1])
            
            rhs_a = self._resolveValues(rhs, True)
            
            if rhs_a.shape[0] == 1:
                return [self._addConstraint(idx, A[i,:], ctype, rhs_a[0])
                        for 0 <= i < A.shape[0]]
            elif rhs_a.shape[0] == A.shape[0]:
                return [self._addConstraint(idx, A[i,:], ctype, rhs_a[i])
                        for 0 <= i < A.shape[0]]
            else:
                raise ValueError("Length of right hand side in constraint must be either 1 or match the number of constraints given.")
        else:
            assert False

    cdef _addConstraintDict(self, dict d, str ctype, rhs):
        I, V = self._getArrayPairFromDict(d)
        return self._addConstraint(I, V, ctype, rhs)
            
    cdef _addConstraintTupleList(self, list l, str ctype, rhs)
        I, V = self._getArrayPairFromTupleList(l)
        return self._addConstraint(I, V, ctype, rhs)


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
        Clears the current objective function.  Note that if
        setObjective() is called with `clear_previous = True`, this
        function is implied.
        """
        
        self._resetObjective()
        self._obj_func_specified = True

    def setObjective(self, coefficients, bint clear_previous=True):
        """
        Sets coefficients in the objective function.  
        """

        # What we do depends on the type
        coefftype = type(coefficients)

        cdef tuple t

        if coefftype is tuple:
            t = <tuple>coefficients
            
            # It's split into index, value pair
            if len(t) != 2:
                raise TypeError("coefficients should be either a single array, "
                                "a dictionary, or a 2-tuple with (index block, values)")

            idx, val = t

        elif coefftype is ndarray:
            idx, val = None, coefficients

        elif coefftype is list:
            # Two possible ways to interpret a list; as a sequence of
            # tuples or as a representation of an array; if it is a
            # sequence of 2-tuples, intepret it this way 
            
            # Test and see if it's a list of sequences
            is_list_sequence = self._isTupleList(<list>coefficients)

            # Test and see if it's a list of scalars
            if is_list_sequence: 
                is_numerical_sequence = False
            else:
                is_numerical_sequence = self._isNumericalList(<list>coefficients)

            if is_list_sequence:
                idx, val = self._getArrayPairFromTupleList(<list>coefficients)
            elif is_numerical_sequence:
                idx, val = None, array(<list>coefficients, dtype=npfloat)
            else:
                raise TypeError("Coefficient list must be either list of scalars or list of 2-tuples.")
                    
        elif coefftype is dict:
            idx, val = self._getArrayPairFromDict(<dict>coefficients)

        else:
            raise TypeError("Type of coefficients not recognized; must be dict, list, 2-tuple, or array.")

        # Now that we've got the idx list and the values, run with it
        if clear_previous:
            self._resetObjective()
        
        self._stackOnInterpretedKeyValuePair(
            self._obj_func_keys, self._obj_func_values, idx, val, &self._obj_func_n_vals_count)

        self._obj_func_specified = True


    cdef tuple _getCurrentObjectiveFunction(self):

        cdef tuple t = self._getArrayPairFromKeyValuePair(
            self._obj_func_keys, 
            self._obj_func_values, 
            self._obj_func_n_vals_count)
        
        self._resetObjective()

        return t


    ############################################################
    # Methods dealing with variable bounds

    cpdef setUnbounded(self, var):
        """
        Sets the variable `var` to unbounded (default is >=0).  This
        is equivalent to setLowerBound(None), setUpperBound(None)
        """
        
        setLowerBound(var, None)
        setUpperBound(var, None)

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
        

    cdef _setBound(self, var, b, bint lower_bound):

        cdef ar ba

        if b is None:
            idx = self._resolveIdxBlock(var, 1)
            val = -infty if lower_bound else infty
        elif type(b) is ndarray:
            ba = b
            if ba.ndim != 1:
                raise TypeError("Bound specification must be either scalar or 1d array.")

            idx = self._resolveIdxBlock(var, ba.shape[0])
            val = b.copy()
        elif isnumeric(b):
            idx = self._resolveIdxBlock(var, 1)
            val = b

        if lower_bound:
            self._stackOnInterpretedKeyValuePair(
                self._lower_bound_keys, self._lower_bound_values, idx, val, &self._lower_bound_count)
        else:
            self._stackOnInterpretedKeyValuePair(
                self._upper_bound_keys, self._upper_bound_values, idx, val, &self._upper_bound_count)
            

    cpdef setUpperBound(self, size_t idx, ub):
        """
        Sets the upper bound of variable idx to ub.  If ub is None,
        then it sets the upper bound to Infinity.
        """

        cdef real lb
        
        if ub is None:  # free it

            lb = get_lowbo(self.lp, idx + 1)

            set_unbounded(self.lp, idx + 1)
            
            if lb != get_infinite(self.lp):
                set_lowbo(self.lp, idx + 1, lb)
        else:
            set_upbo(self.lp, idx + 1, <real?>ub)


    cdef applyVariableBounds(self):
        
        assert self.lp != NULL

        cdef ar[uint_t, mode="c"] I
        cdef ar[double, mode="c"] V
               
        # First the lower bounds; thus we can use set_unbounded on them
        I, V = self._getArrayPairFromKeyValuePair(
            self._lower_bound_keys, self._lower_bound_values, self._lower_bound_count)

        cdef size_t i

        for 0 <= i < I.shape[0]:
            if V[i] == -infty:
                set_unbounded(self.lp, I[i] + 1)
            else:
                set_lowbo(self.lp, I[i] + 1, V[i])

        # Now set the lower bounds; this is trickier as set_unbounded
        # undoes the lower bound

        I, V = self._getArrayPairFromKeyValuePair(
            self._lower_bound_keys, self._lower_bound_values, self._lower_bound_count)

        cdef size_t i

        cdef double lp_infty = get_infinite(self.lp)
        cdef double lb

        for 0 <= i < I.shape[0]:
            if V[i] == infty:
                lb = get_lowbo(self.lp, I[i] + 1)
                set_unbounded(self.lp,  I[i] + 1)
            
                if lb != lp_infty:
                    set_lowbo(self.lp,  I[i] + 1, lb)
            else:
                set_upbo(self.lp, I[i] + 1, V[i])


    ############################################################
    # Helper functions for turning dictionaries or tuple-lists into an
    # index array + value array.

    cdef bint _isTupleList(self, list l):
        for t in l:
            if type(t) is not tuple or len(<tuple>t) != 2:
                return False

        return True
        
    cdef bint _isNumericalList(self, list l):
        for t in l:
            if not isnumeric(t):
                return False

        return True
       
    cdef tuple _getArrayPairFromDict(self, dict d):
        # Builds an array pair from a dictionary

        cdef list key_buf = [], list val_buf = []
        cdef size_t idx_count = 0

        for k, v in d.iteritems():
            self._stackOnInterpretedKeyValuePair(key_buf, val_buf, k, v, &idx_count)

        cdef ar I = empty(idx_count, dtype=uint)
        cdef ar V = empty(idx_count, dtype=npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)

        return (I, V)


    cdef tuple _getArrayPairFromList(self, list l):
        # Builds an array pair from a dictionary

        cdef list key_buf = [], list val_buf = []
        cdef size_t idx_count = 0

        for k, v in l:
            self._stackOnInterpretedKeyValuePair(key_buf, val_buf, k, v, &idx_count)

        cdef ar I = empty(idx_count, dtype=uint)
        cdef ar V = empty(idx_count, dtype=npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)

        return (I, V)

    cdef tuple _getArrayPairFromKeyFaluePair(self, list key_buf, list val_buf, size_t idx_count):
        cdef ar I = empty(idx_count, dtype=uint)
        cdef ar V = empty(idx_count, dtype=npfloat)

        self._fillIndexValueArraysFromIntepretedStack(I, V, key_buf, val_buf)
        
        return (I, V)
    
    cdef _stackOnInterpretedKeyValuePair(self, list key_buf, list val_buf, k, v, size_t *count):
        # Appends interpreted key value pairs to the lists

        cdef ar[size_t] tI
        cdef ar[double] tV
    
        if isnumeric(k) and isnumeric(vo):
            if (<size_t>k) != k:
                raise ValueError("Could not interpret '%s' as nonegative index." % str(k))

            key_buf.append(k)
            val_buf.append(vo)
            count[0] += 1

        elif type(k) is str or type(k) is tuple:
            tV = npfloat(self._resolveValues(vo, True))
            tI = self._resolveIdxBlock(<str>k, tV.shape[0])
            key_buf.append(tI)
            val_buf.append(tV)

            assert not tI is None

            count[0] += tI.shape[0]

        elif k is None:
            tV = npfloat(self._resolveValues(vo, True))
            tI = self._resolveIdxBlock(None, tV.shape[0])
            
            if tI is None: 
                tI = arange(self.n_columns)
                
            val_buf.append(tV)
            count[0] += tI.shape[0]

        else:
            raise TypeError("Error interpreting key/value pair as index/value pair.")


    cdef _fillIndexValueArraysFromIntepretedStack(self, ar I_o, ar V_o, list key_buf, list val_buf):
        
        cdef ar[size_t, mode="c"] I = I_o
        cdef ar[double, mode="c"] V = V_o

        cdef ar[size_t] tI
        cdef ar[double] vI
        
        cdef size_t i, j, ii = 0

        assert len(key_buf) == len(val_buf)

        for 0 <= i < len(key_buf):
            k = key_buf[i]
            v = val_buf[i]

            if isnumeric(idx):
                I[ii] = <size_t>k
                V[ii] = <double>v
                ii += 1
            elif type(k) is ndarray:
                tI = k
                tV = v

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
            else:
                assert False
            
        assert ii == I.shape[0]



    ############################################################
    # Now stuff for solving the model
        
    def solve(self, **options):
        """
        Solves the given model.  


        Basis options
        ==================================================

        To initialize the model with a specific basis, pass ``basis =
        <ndarray>`` as one of the arguments.  The basis must be a
        saved basis from a previous call to getBasis(), on a solved
        model. 
        """

        #set_add_rowmode(self.lp, True)
        #set_add_rowmode(self.lp, False)

        cdef str k

        # Get the current options dict
        cdef dict option_dict = self.getOptions()

        for k, v in options.iteritems():
            option_dict[k.lower()] = v

        # Set any flags having to do with the presolve
        cdef unsigned long n
        cdef unsigned long presolve = 0

        for k, n in _presolve_flags.iteritems():
            if k in option_dict and option_dict[k]:
                presolve += n
        
        set_presolve(self.lp, presolve, 100)

        # Set any flags having to do with the pricing
        if "pricer" in option_dict:
            
            pricer = option_dict["pricer"].lower()

            if type(pricer) is not str or pricer not in _pricer_lookup:
                raise ValueError("pricer option must be one of %s."
                                 % (','.join(["'%s'" % p for p in _pricer_lookup.iterkeys()])))
        else:
            pricer = "devex"
            
        cdef int pricer_option = _pricer_lookup[pricer]

        # Now see if there are other flags in this mix
        for k, n in _pricer_flags.iteritems():
            if k in option_dict and option_dict[k]:
                pricer_option += n

        set_pivoting(self.lp, pricer_option)

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

    cpdef print_lp(self):
        print_lp(self.lp)


    ############################################################
    # Methods for dealing with the constraint buffers

    cdef setConstraint(self, size_t row_idx, ar idx, ar row, str ctypestr, rhs):
        
        # First get the right cstr
        cdef _Constraint* cstr = self.getConstraintStruct(row_idx)

        if cstr == NULL:
            raise MemoryError

        setupConstraint(cstr, row_idx, idx, row, ctypestr, rhs)

    cdef _addConstraint(self, ar idx, ar row, str ctypestr, rhs):
        cdef size_t row_idx = self.n_rows
        self.setConstraint(row_idx, idx, row, ctypestr, rhs)
        return row_idx

    cdef _Constraint* getConstraintStruct(self, size_t row_idx):
        
        # First see if our double-buffer thing is ready to go
        cdef size_t buf_idx = <size_t>(row_idx // cStructBufferSize)
        cdef size_t idx     = <size_t>(row_idx % cStructBufferSize)
        cdef size_t new_size

        cdef _Constraint* buf

        # Ensure proper sizing of constraint buffer
        if buf_idx >= self.current_c_buffer_size:
            
            # Be agressive, since we anticipate growing incrementally,
            # and each one is a lot of constraints

            new_size = 128 + 2*buf_idx
            new_size -= (new_size % 128)

            if self._c_buffer == NULL:
                self._c_buffer = <_Constraint**>malloc(new_size*sizeof(_Constraint*))
            else:
                self._c_buffer = <_Constraint**>realloc(self._c_buffer, new*sizeof(_Constraint*))
            
            if self._c_buffer == NULL:
                return NULL  

            memset(&self._c_buffer[self.current_c_buffer_size-1], 0,
                    (buf_idx - self.current_c_buffer_size)*sizeof(_Constraint*))

            self.current_c_buffer_size = new_size

        # Now make sure that the buffer is ready
        buf = self._c_buffer[buf_idx]
        
        if buf == NULL:
            buf = self._c_buffer[buf_idx] = \
                malloc(cStructBufferSize*sizeof(_Constraint))
            
            if buf == NULL:
                return NULL

            memset(buf, 0, cStructBufferSize*sizeof(_Constraint))

        # Now finally this determines the new model size

        if row_idx >= self.n_rows:
            self.n_rows = row_idx + 1
        
        return &buf[idx]

    cdef applyAllConstraints(self):
    
        assert self.lp != NULL

        cdef size_t i, j
        cdef _Constraint *buf

        # This enables the slicing to work properly
        cdef ar[int, mode="c"] countrange = arange(self.n_columns)
        cdef int* cr_ptr = <int*>countrange.data

        for 0 <= i < self.current_c_buffer_size:
            buf = self._c_buffer[i]
        
            if buf == NULL:
                continue

            for 0 <= j < cStructBufferSize:
                if inUse(buf[j]):
                    setInLP(buf[j], self.lp, self.n_columns, cr_ptr)

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
