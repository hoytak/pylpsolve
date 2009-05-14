import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import ndarray as ar, ones, eye

class TestLPProblem(unittest.TestCase):

    def test01_basic(self):
        # test singleton
        
        lp = LPSolve()

        lp.addConstraint( [1], ">", 1)
        lp.setObjective( [1], mode = "minimize")

        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 1)
        self.assert_(v[0] == 1)


    # Test retrieval of blocks
    def test02_blocks_01(self):
        lp = LPSolve()

        self.assert_(lp.getVariables(2, "a1") == (0,2))

    # Test retrieval of blocks
    def test02_blocks_01_r(self):
        lp = LPSolve()

        self.assert_(lp.getVariables("a1", 2) == (0,2))

    def test02_blocks_02(self):
        lp = LPSolve()

        self.assert_(lp.getVariables(2, "a1") == (0,2))
        self.assert_(lp.getVariables(4, "a2") == (2,6))
        self.assert_(lp.getVariables("a1") == (0,2))

    def test02_blocks_02_reverse_order(self):
        lp = LPSolve()

        self.assert_(lp.getVariables("a1", 2) == (0,2))
        self.assert_(lp.getVariables("a2", 4) == (2,6))
        self.assert_(lp.getVariables("a1", 2) == (0,2))

    def test02_blocks_02_reverse_order_mixed(self):
        lp = LPSolve()

        self.assert_(lp.getVariables("a1", 2) == (0,2))
        self.assert_(lp.getVariables(4, "a2") == (2,6))
        self.assert_(lp.getVariables("a1", 2) == (0,2))

    def test02_blocks_03_bad_recall(self):
        lp = LPSolve()

        self.assert_(lp.getVariables(2, "a1") == (0,2))
        self.assert_(lp.getVariables(4, "a2") == (2,6))
        self.assertRaises(ValueError, lambda: lp.getVariables(3, "a1"))

    def test02_blocks_04_bad_size_01(self):
        lp = LPSolve()
        
        self.assertRaises(ValueError, lambda: lp.getVariables("a1", 0))

    def test02_blocks_04_bad_size_02(self):
        lp = LPSolve()
        
        self.assertRaises(ValueError, lambda: lp.getVariables("a1", 0.5))

    def test02_blocks_04_bad_size_03(self):
        lp = LPSolve()
        
        self.assertRaises(ValueError, lambda: lp.getVariables("a1", -1))

    def test02_blocks_04_bad_size_04(self):
        lp = LPSolve()
        
        self.assertRaises(ValueError, lambda: lp.getVariables("a1", "a2"))

    def test02_blocks_04_bad_size_05(self):
        lp = LPSolve()
        
        self.assertRaises(ValueError, lambda: lp.getVariables(0, 2))


    # test constraint adding by (name, value array)
    #def test03_constraints_01_explicit_blocks(self):
    #    lp = LPSolve()

    #b1 = lp.getVariables(1, "a1")
    #    b2 = lp.
        

    # test constraint adding by (index array, value array)
    # test constraint adding by (index array, wrong typed value array)
    # test constraint adding by (wrong typed index array, value array)
    # test constraint adding by (index array, list)
    # test constraint adding by (name, list)
    # test constraint adding by (tuple, list)
    # test constraint adding absolute array
    # test constraint adding absolute list
    # test constraint adding absolute array of wrong types
    # test constraint adding absolute array of double types
    # test constraint adding by list of tuples
    # test constraint adding by dict
    # test constraint adding by dict with tuples:vectors
    # test constraint adding by dict with names:vectors
    # test constraint adding by dict with tuples:scalars

    # test constraint adding by dict with names:scalars, names
    # previously defined

    # test constraint adding by dict with tuples:scalars, names
    # previously defined

    # test constraint adding by dict with tuples:scalars, names
    # not previously defined

    # test constraint adding with a 2d array with scalar b
    # test constraint adding with a 2d array with vector b
    # test constraint adding with a list of 1d arrays with scalar b
    # test constraint adding with a list of 1d arrays with vector b
    # test constraint adding with a list of lists with scalar b
    # test constraint adding with a list of lists with vector b

    # test constraint adding with a tuple + 2d array with scalar b
    # test constraint adding with a tuple + 2d array with vector b
    # test constraint adding with a tuple + list of 1d arrays with scalar b
    # test constraint adding with a tuple + list of 1d arrays with vector b
    # test constraint adding with a tuple + list of lists with scalar b
    # test constraint adding with a tuple + list of lists with vector b

    # test constraint adding with a name + 2d array with scalar b
    # test constraint adding with a name + 2d array with vector b
    # test constraint adding with a name + list of 1d arrays with scalar b
    # test constraint adding with a name + list of 1d arrays with vector b
    # test constraint adding with a name + list of lists with scalar b
    # test constraint adding with a name + list of lists with vector b

    # test constraint adding with scalar multiple on previously
    # defined variable group
        
    # test constraint adding with scalar multiple with tuple indexing
    # on group that's not previously defined.

    # test that running twice gives same answer

    # test that running with maximize, then minimize works with no
    # other changes

    # test that guesses work

    
    



if __name__ == '__main__':
    unittest.main()

