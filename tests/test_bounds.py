import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array as ar, ones, eye, float64, uint, inf

class TestBounds(unittest.TestCase):
    
    def checkLB(self, opts, lb):

        # these are indices to bound
        indices = {}
        indices["t"] = (0,3)
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2],dtype=uint)
        indices["f"] = ar([0,1,2],dtype=float64)

        lbvalues = {}
        lbvalues["s"] = lb
        lbvalues["l"] = [lb, lb, lb]
        lbvalues["a"] = ar([lb, lb, lb])

        lp = LPSolve()

        if opts[0] == "N":
            lp.getVariables(indices["N"], 3)

        lp.setObjective([1,1,1,1,1,1])

        lp.setLowerBound(indices[opts[0]], lbvalues[opts[1]])

        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), lb*3)

            v = lp.getSolution()

            self.assert_(len(v) == 6)
            self.assertAlmostEqual(v[0], lb)
            self.assertAlmostEqual(v[1], lb)
            self.assertAlmostEqual(v[2], lb) 

            self.assertAlmostEqual(v[3], 0)
            self.assertAlmostEqual(v[4], 0)
            self.assertAlmostEqual(v[5], 0) 
           
    def testLB_neg_ts(self): self.checkLB("ts", -12.34)
    def testLB_neg_Ns(self): self.checkLB("Ns", -12.34)
    def testLB_neg_ls(self): self.checkLB("ls", -12.34)
    def testLB_neg_as(self): self.checkLB("as", -12.34)
    def testLB_neg_fs(self): self.checkLB("fs", -12.34)

    def testLB_pos_ts(self): self.checkLB("ts", 12.34)
    def testLB_pos_Ns(self): self.checkLB("Ns", 12.34)
    def testLB_pos_ls(self): self.checkLB("ls", 12.34)
    def testLB_pos_as(self): self.checkLB("as", 12.34)
    def testLB_pos_fs(self): self.checkLB("fs", 12.34)

    def testLB_neg_tl(self): self.checkLB("tl", -12.34)
    def testLB_neg_Nl(self): self.checkLB("Nl", -12.34)
    def testLB_neg_ll(self): self.checkLB("ll", -12.34)
    def testLB_neg_al(self): self.checkLB("al", -12.34)
    def testLB_neg_fl(self): self.checkLB("fl", -12.34)

    def testLB_pos_tl(self): self.checkLB("tl", 12.34)
    def testLB_pos_Nl(self): self.checkLB("Nl", 12.34)
    def testLB_pos_ll(self): self.checkLB("ll", 12.34)
    def testLB_pos_al(self): self.checkLB("al", 12.34)
    def testLB_pos_fl(self): self.checkLB("fl", 12.34)

    def testLB_neg_ta(self): self.checkLB("ta", -12.34)
    def testLB_neg_Na(self): self.checkLB("Na", -12.34)
    def testLB_neg_la(self): self.checkLB("la", -12.34)
    def testLB_neg_aa(self): self.checkLB("aa", -12.34)
    def testLB_neg_fa(self): self.checkLB("fa", -12.34)

    def testLB_pos_ta(self): self.checkLB("ta", 12.34)
    def testLB_pos_Na(self): self.checkLB("Na", 12.34)
    def testLB_pos_la(self): self.checkLB("la", 12.34)
    def testLB_pos_aa(self): self.checkLB("aa", 12.34)
    def testLB_pos_fa(self): self.checkLB("fa", 12.34)


    def checkUB(self, opts, ub):

        # these are indices to bound
        indices = {}
        indices["t"] = (0,3)
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2],dtype=uint)
        indices["f"] = ar([0,1,2],dtype=float64)

        ubvalues = {}
        ubvalues["s"] = ub
        ubvalues["l"] = [ub, ub, ub]
        ubvalues["a"] = ar([ub, ub, ub])

        lp = LPSolve()

        if opts[0] == "N":
            lp.getVariables(indices["N"], 3)

        lp.setObjective([1,1,1,1,1,1])
        lp.addConstraint( ((3,6), [[1,0,0],[0,1,0],[0,0,1]]), "<=", 10)
        lp.setMaximize()

        lp.setLowerBound(indices[opts[0]], None)
        lp.setUpperBound(indices[opts[0]], ubvalues[opts[1]])

        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), ub*3 + 10 * 3)

            v = lp.getSolution()

            self.assert_(len(v) == 6)
            self.assertAlmostEqual(v[0], ub)
            self.assertAlmostEqual(v[1], ub)
            self.assertAlmostEqual(v[2], ub) 

            self.assertAlmostEqual(v[3], 10)
            self.assertAlmostEqual(v[4], 10)
            self.assertAlmostEqual(v[5], 10) 

    def testUB_neg_ts(self): self.checkUB("ts", -12.34)
    def testUB_neg_Ns(self): self.checkUB("Ns", -12.34)
    def testUB_neg_ls(self): self.checkUB("ls", -12.34)
    def testUB_neg_as(self): self.checkUB("as", -12.34)
    def testUB_neg_fs(self): self.checkUB("fs", -12.34)

    def testUB_pos_ts(self): self.checkUB("ts", 12.34)
    def testUB_pos_Ns(self): self.checkUB("Ns", 12.34)
    def testUB_pos_ls(self): self.checkUB("ls", 12.34)
    def testUB_pos_as(self): self.checkUB("as", 12.34)
    def testUB_pos_fs(self): self.checkUB("fs", 12.34)

    def testUB_neg_tl(self): self.checkUB("tl", -12.34)
    def testUB_neg_Nl(self): self.checkUB("Nl", -12.34)
    def testUB_neg_ll(self): self.checkUB("ll", -12.34)
    def testUB_neg_al(self): self.checkUB("al", -12.34)
    def testUB_neg_fl(self): self.checkUB("fl", -12.34)

    def testUB_pos_tl(self): self.checkUB("tl", 12.34)
    def testUB_pos_Nl(self): self.checkUB("Nl", 12.34)
    def testUB_pos_ll(self): self.checkUB("ll", 12.34)
    def testUB_pos_al(self): self.checkUB("al", 12.34)
    def testUB_pos_fl(self): self.checkUB("fl", 12.34)

    def testUB_neg_ta(self): self.checkUB("ta", -12.34)
    def testUB_neg_Na(self): self.checkUB("Na", -12.34)
    def testUB_neg_la(self): self.checkUB("la", -12.34)
    def testUB_neg_aa(self): self.checkUB("aa", -12.34)
    def testUB_neg_fa(self): self.checkUB("fa", -12.34)

    def testUB_pos_ta(self): self.checkUB("ta", 12.34)
    def testUB_pos_Na(self): self.checkUB("Na", 12.34)
    def testUB_pos_la(self): self.checkUB("la", 12.34)
    def testUB_pos_aa(self): self.checkUB("aa", 12.34)
    def testUB_pos_fa(self): self.checkUB("fa", 12.34)


    def checkLBUBMix(self, opts, lb, ub):

        # these are indices to bound
        lbindices = (0,3)

        ubindices = {}
        ubindices["t"] = (3,6)
        ubindices["n"] = "a"
        ubindices["N"] = "a"
        ubindices["l"] = [3,4,5]
        ubindices["a"] = ar([3,4,5],dtype=uint)
        ubindices["f"] = ar([3,4,5],dtype=float64)

        ubvalues = {}
        ubvalues["s"] = ub
        ubvalues["l"] = [ub, ub, ub]
        ubvalues["a"] = ar([ub, ub, ub])

        lbvalues = {}
        lbvalues["s"] = lb
        lbvalues["l"] = [lb, lb, lb]
        lbvalues["a"] = ar([lb, lb, lb])

        lp = LPSolve()

        lp.setLowerBound(lbindices, lbvalues[opts[1]])

        if opts[0] == "N":
            lp.getVariables(ubindices["N"], 3)

        lp.setUpperBound(ubindices[opts[0]], ubvalues[opts[1]])

        lp.setObjective([1,1,1,-1,-1,-1])


        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), lb*3 - ub*3)

            v = lp.getSolution()

            self.assert_(len(v) == 6)
            self.assertAlmostEqual(v[0], lb)
            self.assertAlmostEqual(v[1], lb)
            self.assertAlmostEqual(v[2], lb) 

            self.assertAlmostEqual(v[3], ub)
            self.assertAlmostEqual(v[4], ub)
            self.assertAlmostEqual(v[5], ub)

    def testLBUBMix_neg_ts(self): self.checkLBUBMix("ts", -12.34, 5.3)
    def testLBUBMix_neg_Ns(self): self.checkLBUBMix("Ns", -12.34, 5.3)
    def testLBUBMix_neg_ls(self): self.checkLBUBMix("ls", -12.34, 5.3)
    def testLBUBMix_neg_as(self): self.checkLBUBMix("as", -12.34, 5.3)
    def testLBUBMix_neg_fs(self): self.checkLBUBMix("fs", -12.34, 5.3)

    def testLBUBMix_pos_ts(self): self.checkLBUBMix("ts", 12.34, 5.3)
    def testLBUBMix_pos_Ns(self): self.checkLBUBMix("Ns", 12.34, 5.3)
    def testLBUBMix_pos_ls(self): self.checkLBUBMix("ls", 12.34, 5.3)
    def testLBUBMix_pos_as(self): self.checkLBUBMix("as", 12.34, 5.3)
    def testLBUBMix_pos_fs(self): self.checkLBUBMix("fs", 12.34, 5.3)

    def testLBUBMix_neg_tl(self): self.checkLBUBMix("tl", -12.34, 5.3)
    def testLBUBMix_neg_Nl(self): self.checkLBUBMix("Nl", -12.34, 5.3)
    def testLBUBMix_neg_ll(self): self.checkLBUBMix("ll", -12.34, 5.3)
    def testLBUBMix_neg_al(self): self.checkLBUBMix("al", -12.34, 5.3)
    def testLBUBMix_neg_fl(self): self.checkLBUBMix("fl", -12.34, 5.3)

    def testLBUBMix_pos_tl(self): self.checkLBUBMix("tl", 12.34, 5.3)
    def testLBUBMix_pos_Nl(self): self.checkLBUBMix("Nl", 12.34, 5.3)
    def testLBUBMix_pos_ll(self): self.checkLBUBMix("ll", 12.34, 5.3)
    def testLBUBMix_pos_al(self): self.checkLBUBMix("al", 12.34, 5.3)
    def testLBUBMix_pos_fl(self): self.checkLBUBMix("fl", 12.34, 5.3)

    def testLBUBMix_neg_ta(self): self.checkLBUBMix("ta", -12.34, 5.3)
    def testLBUBMix_neg_Na(self): self.checkLBUBMix("Na", -12.34, 5.3)
    def testLBUBMix_neg_la(self): self.checkLBUBMix("la", -12.34, 5.3)
    def testLBUBMix_neg_aa(self): self.checkLBUBMix("aa", -12.34, 5.3)
    def testLBUBMix_neg_fa(self): self.checkLBUBMix("fa", -12.34, 5.3)

    def testLBUBMix_pos_ta(self): self.checkLBUBMix("ta", 12.34, 5.3)
    def testLBUBMix_pos_Na(self): self.checkLBUBMix("Na", 12.34, 5.3)
    def testLBUBMix_pos_la(self): self.checkLBUBMix("la", 12.34, 5.3)
    def testLBUBMix_pos_aa(self): self.checkLBUBMix("aa", 12.34, 5.3)
    def testLBUBMix_pos_fa(self): self.checkLBUBMix("fa", 12.34, 5.3)


if __name__ == '__main__':
    unittest.main()

