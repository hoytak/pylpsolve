import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LP, LPException
from numpy import array as ar, ones, eye, float64, uint

class TestBases(unittest.TestCase):
    
    def testBasicBasis(self):
        
        # this should work as it's in the examples

        lp = LP()

        lp.addConstraint( (0, 1), "<", 3)
        lp.addConstraint( (1, 1), "<", 3)

        lp.setMaximize()

        lp.setObjective([1,1])

        lp.solve(guess = [3,3])

        self.assert_(lp.getInfo("Iterations") == 0, lp.getInfo("Iterations"))

    def checkBasisRecycling(self, opts, constraint_arg_list, objective):

        def getLP():

            lp = LP()

            for c, t, b in constraint_arg_list:
                lp.addConstraint(c,t,b)

            lp.setObjective(objective)

            return lp
    
        lp1 = getLP()

        #lp1.print_lp()

        lp1.solve()

        if opts[0] == "f":
            full_basis = True
        elif opts[0] == "p":
            full_basis = False
        else:
            assert False

        #print "solution = ", lp1.getSolution()
        #print "basis = ", lp1.getBasis()

        lp2 = getLP()

        if opts[1] == "b":
            lp2.solve(basis = lp1.getBasis(full_basis))
        elif opts[1] == "g":
            lp2.solve(guess = lp1.getSolution() )
        elif opts[1] == "B":
            lp2.solve(guess = lp1.getSolution(), basis = lp1.getBasis(full_basis) )
        else:
            assert False

        n_it_1 = lp1.getInfo("iterations")
        n_it_2 = lp2.getInfo("iterations")

        self.assert_(n_it_2 in [0,1] and (n_it_2 < n_it_1 or n_it_1 == 0),
                     "n_iterations 1 = %d, n_iterations_2 = %d" % (n_it_1, n_it_2))

    lp_01 = [( ([0,1], [1,1]), ">", 2),
             ( ([1,2], [1,1]), ">", 2)]
    lp_01_obj = [1,1,1]
             
    def testBasis_01_fb(self): self.checkBasisRecycling("fb", self.lp_01, self.lp_01_obj)
    def testBasis_01_fg(self): self.checkBasisRecycling("fg", self.lp_01, self.lp_01_obj)
    def testBasis_01_fB(self): self.checkBasisRecycling("fB", self.lp_01, self.lp_01_obj)
    def testBasis_01_pb(self): self.checkBasisRecycling("pb", self.lp_01, self.lp_01_obj)
    def testBasis_01_pB(self): self.checkBasisRecycling("pB", self.lp_01, self.lp_01_obj)


    lp_02 = [( ([0,1], [2,3]), ">", 5),
             ( ([1,2], [2,3]), ">", 5),
             ( ([2,3], [2,3]), ">", 5),
             ( ([3,4], [2,3]), ">", 5)]
    lp_02_obj = [1,1,1,1,1]

    def testBasis_02_fb(self): self.checkBasisRecycling("fb", self.lp_02, self.lp_02_obj)
    def testBasis_02_fg(self): self.checkBasisRecycling("fg", self.lp_02, self.lp_02_obj)
    def testBasis_02_fB(self): self.checkBasisRecycling("fB", self.lp_02, self.lp_02_obj)
    def testBasis_02_pb(self): self.checkBasisRecycling("pb", self.lp_02, self.lp_02_obj)
    def testBasis_02_pB(self): self.checkBasisRecycling("pB", self.lp_02, self.lp_02_obj)



if __name__ == '__main__':
    unittest.main()

