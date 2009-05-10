from numpy cimport ndarray as ar
from numpy import empty, ones, zeros, uint, arange

from constraintstruct cimport _Constraint, setupConstraint, \
    setInLP, clearConstraint, isInUse

import warnings

cimport cython

ctypedef double real
ctypedef unsigned char ecode

import optionlookup

######################################################################
# A few early binding things

cdef dict default_options = optionlookup._default_options
cdef dict presolve_flags  = optionlookup._presolve_flags
cdef dict pricer_lookup   = optionlookup._pricer_lookup
cdef dict pricer_flags    = optionlookup._pricer_flags


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


class LPSolveException(Exception): pass
        
################################################################################
# Now the full class

DEF m_NewModel      = 0
DEF m_UpdatingModel = 1

# This is the size of the constraint buffer blocks
DEF cStructBufferSize = 128  

cdef class LPSolve:
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
    cdef dict _var_bounds
    
    # The objective function
    cdef list _obj_func_values

    # Methods relating to named variable group
    cdef dict _named_index_blocks

    def __cinit__(self):

        self.lp = NULL
        self.options = default_options.copy()

        self.current_c_buffer_size = 0
        self._c_buffer = NULL

        self._var_bounds = {}
        self._obj_func_values = []
        self._named_index_blocks = {}

    def __dealloc__(self):
        self.clearConstraintBuffers()
        
        if self.lp != NULL:
            delete_lp(self.lp)

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

    cpdef ar getVariableIndexBlock(self, size_t size, str name = None):
        """
        Returns a new or current block of 'size' variable indices to
        be used in the current LP.
        
        If `name` is not None, then other functions
        (e.g. addConstraint) can accept this accept this string as the
        index argument.

        # Put in use cases

        """

        cdef ar idx

        if name is None:
            idx = uint(arange(self.n_columns, self.n_columns + size))
            self.n_columns += size
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
                self.n_columns += size
                self._named_index_blocks[name] = idx
                return idx
                
    def getVariableIndex(self, str name = None):
        """
        Returns a single unused index (unless name refers to a
        previously named index).
        """

        if name is None:
            self.n_columns += 1
            return self.n_columns -1
        else:
            return self.getVariableIndexBlock(1, name)[0]

    ############################################################
    # Methods dealing with constraints

    cpdef addConstraint(self, coefficients, str constraint_type, rhs):
        """        
        Adds a constraint, or set of constraints to the lp.

        `coefficients` may be either a single array, a dictionary, or
        a 2-tuple with the form (index block, value array).  In the
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

        cdef ar idx, vals

        cdef size_t size 

        if type(coefficients) is tuple or type(coefficients) is list:

            if len(coefficients) != 2:
                raise TypeError("coefficients should be either a single array, "
                                "a dictionary, or a 2-tuple with (index block, values)")

            t_idx, t_vals = coefficients

            


            if type(t_idx) is 




        try:
            constraint_id = _constraint_map[constraint_type]
        except KeyError:
            try:
                constraint_id = _constraint_map[constraint_type.lower()]
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

                self.copyIntoIndices_shift(col_idx)
                self._copy_into_real_buffer(row)
                self._addSparseConstraintFromBuffer(n, constraint_id, rhs)

            elif row.ndim == 2:

                if col_idx.ndim == 1:
                    n = col_idx.shape[0]

                    if row.shape[1] != col_idx.shape[0]:
                        raise LPSolveException("Length of index array (%d) must match number of coefficients (%d)."
                                                % (col_idx.shape[0], row.shape[1]))

                    self.copyIntoIndices_shift(col_idx)
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
                        self.copyIntoIndices_shift(col_idx[i,:])
                        self._copy_into_real_buffer(row[i,:])
                        self._addSparseConstraintFromBuffer(n, constraint_id, bcast[i])
                else:
                    raise LPSolveException("Column index array must be either 1d or 2d.")
                    
            else:
                raise LPSolveException("Coefficient matrix in sparse must be either 1d or 2d.")
        else:
            raise LPSolveException("coefficients argument must either be a 1d array, 2-tuple, or dict.")


        

    ############################################################
    # methods dealing with the objective function

    def clearObjective(self):
        """
        Clears the currnet objective function.  Because the 

        """
        
        self._obj_func_values = []

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
        cdef size_t n, i, k, s
        cdef bint sizing_okay

        if isinstance(coefficients, ar):
            a = coefficients
            s = a.size

            if a.ndim != 1:
                
                sizing_okay = False

                for 0 <= i < a.ndim:
                    if a.size == a.shape[i]:
                        a = a.ravel()
                        sizing_okay = True
                        break

                if not sizing_okay:
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
            self.copyIntoIndices_shift(idx)

            if set_obj_fnex(self.lp, n, self.rbuf, self.intbuf) != 1:
                raise LPSolveException("Error adding objective function.")
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

        if t == 0:
            return t

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


    cpdef setLowerBound(self, size_t idx, lb):
        """
        Sets the lower bound of variable idx to lb.  If lb is None,
        then it sets the lower bound to -Infinity.
        """
        cdef real ub
        
        if lb is None:  # free it

            ub = get_upbo(self.lp, idx + 1)

            set_unbounded(self.lp, idx + 1)
            
            if ub != get_infinite(self.lp):
                set_upbo(self.lp, idx + 1, ub)
        else:
            set_lowbo(self.lp, idx + 1, <real?>lb)

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
        Solves the given model.  


        Basis options
        ==================================================

        To initialize the model with a specific basis, pass ``basis =
        <ndarray>`` as one of the arguments.  The basis must be a
        saved basis from a previous call to getBasis(), on a solved
        model. 
        """

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

    cdef addConstraint(self, ar idx, ar row, str ctypestr, rhs):
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

        # Now make sure that buffer is ready
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
