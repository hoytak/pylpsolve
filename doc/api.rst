PyLPSolve Reference
========================================

.. contents::		       

Construction
----------------------------------------

.. method:: pylpsolve.LP.__init__(self)

    Create a new LP instance.  All construction and destruction
    routines in the low-level code is handled automatically.

.. automethod:: pylpsolve.LP.clear(self)

Index Blocks
----------------------------------------

.. automethod:: pylpsolve.LP.getIndexBlock(self, a1, a2 = None)


Constraints
----------------------------------------

.. automethod:: pylpsolve.LP.addConstraint(self, coefficients, ctypestr, rhs)

.. automethod:: pylpsolve.LP.bindEach(self, indices_1, ctypestr, indices_2)

.. automethod:: pylpsolve.LP.bindSandwich(self, constrained_indices, sandwich_indices)


Variables
----------------------------------------

.. automethod:: pylpsolve.LP.setUnbounded(self, indices)

.. automethod:: pylpsolve.LP.setLowerBound(self, indices, lb)

.. automethod:: pylpsolve.LP.setUpperBound(self, indices, ub)

Objective
----------------------------------------

.. automethod:: pylpsolve.LP.setObjective(self, coefficients, mode = None)

.. automethod:: pylpsolve.LP.addToObjective(self, coefficients)

.. automethod:: pylpsolve.LP.clearObjective(self)

.. automethod:: pylpsolve.LP.setMinimize(self, minimize=True)

.. automethod:: pylpsolve.LP.setMaximize(self, maximize=True)

Options
----------------------------------------

.. automethod:: pylpsolve.LP.setOption(self, *args, **kw_options)

.. automethod:: pylpsolve.LP.getOption(self, option)

.. automethod:: pylpsolve.LP.getOptionDict(self)


Solving and Solution Values
----------------------------------------

.. automethod:: pylpsolve.LP.solve(self, **options)

.. automethod:: pylpsolve.LP.getInfo(self, info)

.. automethod:: pylpsolve.LP.getSolution(self, indices = None)

.. automethod:: pylpsolve.LP.getSolutionDict(self)

.. automethod:: pylpsolve.LP.getObjectiveValue(self)

.. automethod:: pylpsolve.LP.getBasis(self, include_dual_basis = True)
