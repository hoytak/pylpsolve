import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array as ar, ones, eye, float64, uint

class TestErrorCatch(unittest.TestCase):
    # test constraint adding by (wrong typed index array, value array)
    def test01_constraint_rejects_float_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.addConstraint( (ar([0, 1.1, 2],dtype=float64), ar([1,1,1],dtype=float64) ), ">=", 1))

    # test constraint adding by (wrong typed index array, value array)
    def test08_objfunc_rejects_float_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.setObjective( (ar([0, 1.1, 2],dtype=float64), ar([1,1,1],dtype=float64) )))
