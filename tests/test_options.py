import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import ndarray as ar, ones, eye

class TestOptions(unittest.TestCase):
    
    def testOptionRetrieval01(self):

        lp = LPSolve()

        self.assert_(lp.getOption("presolve_rows") == False)

        lp.setOption("presolve_rows", True)
        
        self.assert_(lp.getOption("presolve_rows") == True)
        self.assert_(lp.getOptionDict()["presolve_rows"] == True)

    def testOptionRetrieval02(self):

        lp = LPSolve()

        self.assert_(lp.getOption("presolve_rows") == False)

        lp.setOption(presolve_rows = True)
        
        self.assert_(lp.getOption("presolve_rows") == True)
        self.assert_(lp.getOptionDict()["presolve_rows"] == True)

    def testOptionRetrieval03(self):

        lp = LPSolve()

        self.assert_(lp.getOption("presolve_rows") == False)

        lp.setOption("presolve_cols", True, presolve_rows = True)
        
        self.assert_(lp.getOption("presolve_rows") == True)
        self.assert_(lp.getOptionDict()["presolve_rows"] == True)

        self.assert_(lp.getOption("presolve_cols") == True)
        self.assert_(lp.getOptionDict()["presolve_cols"] == True)

    def testOptionRetrieval_BadValues_01(self):

        lp = LPSolve()

        self.assert_(lp.getOption("presolve_rows") == False)

        self.assertRaises(ValueError,
                          lambda: lp.setOption("presolve_rows", True, bad_option = None))
        
        self.assert_(lp.getOption("presolve_rows") == False)
        self.assert_(lp.getOptionDict()["presolve_rows"] == False)

    def testOptionRetrieval_BadValues_02(self):

        lp = LPSolve()

        self.assert_(lp.getOption("presolve_rows") == False)

        self.assertRaises(TypeError,
                          lambda: lp.setOption(None, True, presolve_rows = True))
        
        self.assert_(lp.getOption("presolve_rows") == False)
        self.assert_(lp.getOptionDict()["presolve_rows"] == False)


if __name__ == '__main__':
    unittest.main()

