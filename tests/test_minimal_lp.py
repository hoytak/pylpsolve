import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array as ar, ones, eye

class TestMinimal(unittest.TestCase):


    def test01_basic_full(self):
        # test singleton
        
        lp = LPSolve()

        lp.addConstraint( [1], ">", 1)
        lp.setObjective( [1], mode = "minimize")

        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 1)
        self.assert_(v[0] == 1)

    def test01_basic_partial(self):
        # test singleton
        
        lp = LPSolve()
        
        lp.addConstraint( (ar([0]), ar([1])), ">", 1)
        print "adding objective"

        lp.setObjective( [1], mode = "minimize")

        print "solving"

        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 1)
        self.assert_(v[0] == 1)


    def test01_basic_secondcol(self):
        # test singleton
        
        lp = LPSolve()

        lp.addConstraint( ([1], [1]), ">", 1)
        lp.setObjective( ([1], [1]), mode = "minimize")

        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 2)
        self.assert_(v[1] == 1)


    # test constraint adding by (name, list)
    def test02_constraints_01_lists_explicit_blocks(self):
        lp = LPSolve()
        
        lp.getVariables("a1", 3)
        lp.addConstraint(("a1", [1,1,1]), ">=", 1)
        lp.setObjective(("a1", [1,2,3]))
        
        try:
            lp.solve()
        except Exception, e:
            lp.print_lp()
            raise e

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 3)
        self.assert_(v[0] == 1)
        self.assert_(v[1] == 0)
        self.assert_(v[2] == 0)

    # test constraint adding by (name, array)
    def test02_constraints_02_arrays_explicit_blocks(self):
        lp = LPSolve()
        
        lp.getVariables("a1", 3)
        lp.addConstraint(("a1", ar([1,1,1])), ">=", 1)
        lp.setObjective("a1", ar([1,2,3]))
        
        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 3)
        self.assert_(v[0] == 1)
        self.assert_(v[1] == 0)
        self.assert_(v[2] == 0)
        
    # test constraint adding by (name, list)
    def test02_constraints_03_lists(self):
        lp = LPSolve()
        
        #lp.getVariables("a1", 3) # not explicit here
        lp.addConstraint(("a1", [1,1,1]), ">=", 1)
        lp.setObjective("a1", [1,2,3])
        
        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 3)
        self.assert_(v[0] == 1)
        self.assert_(v[1] == 0)
        self.assert_(v[2] == 0)

    # test constraint adding by (name, array)
    def test02_constraints_04_arrays(self):
        lp = LPSolve()
        
        #lp.getVariables("a1", 3) # not explicit here
        lp.addConstraint(("a1", ar([1,1,1])), ">=", 1)
        lp.setObjective("a1", ar([1,2,3]))
        
        lp.solve()

        self.assert_(lp.getObjectiveValue() == 1)

        v = lp.getSolution()

        self.assert_(len(v) == 3)
        self.assert_(v[0] == 1)
        self.assert_(v[1] == 0)
        self.assert_(v[2] == 0)
        
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

