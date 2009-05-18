import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array, ones, eye, float64, uint, zeros

from pylpsolve.graphs import graphCut, maximimzeGraphPotential

class TestGraphCuts(unittest.TestCase):

    def checkCut(self, d, source, sink, answer, opts):

        s_edges = [i for i,j in d.iterkeys()]
        d_edges = [j for i,j in d.iterkeys()]
        cap     = d.values()
        
        nV = max(max(s_edges), max(d_edges)) + 1

        def makeArray():
            A = zeros( (nV, nV), dtype=float64)

            for (i, j), c in d.iteritems():
                A[i,j] = c

            return A

        if opts == "A":
            # Check the array version
            ret = graphCut(makeArray(), source, sink)
        elif opts == "L":
            ret = graphCut([[a for a in l] for l in makeArray()], source, sink)
        elif opts == "d":
            ret = graphCut(d, source, sink)
        elif opts == "tl":
            ret = graphCut( (s_edges, d_edges, cap), source, sink)
        elif opts == "ta":
            ret = graphCut( (array(s_edges), array(d_edges), array(cap)), source, sink)
        elif opts == "tfa":
            ret = graphCut( (array(s_edges,dtype=float64), array(d_edges, dtype=float64), array(cap,dtype=float64)), source, sink)
        else:
            assert False

        self.assert_(len(ret) == len(answer))

        for re, ae in zip(ret, answer):
            self.assert_(re == ae, "%s != %s (true)" % (str(ret), str(answer)))

    graph_01 = {
        (0,1) : 4,
        (0,2) : 3,
        (1,2) : 1,
        (1,3) : 1,
        (1,3) : 1,
        (2,4) : 3,
        (3,5) : 2,
        (4,5) : 5}

    answer_01 = [1,1,1,0,0,0]

    def testGraphCut01_A(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "A")
    def testGraphCut01_L(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "L")
    def testGraphCut01_d(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "d")
    def testGraphCut01_tl(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "tl")
    def testGraphCut01_ta(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "ta")
    def testGraphCut01_tfa(self): self.checkCut(self.graph_01, 0, 5, self.answer_01, "tfa")


    graph_02 = {
        (0,1) : 4,
        (0,2) : 3,
        (1,2) : 1,
        (1,3) : 1,
        (1,3) : 1,
        (2,2) : 4,
        (2,4) : 3,
        (3,5) : 2,
        (4,5) : 5,
        (5,5) : 10,
        (6,6) : 1,
        (6,7) : 2,
        (7,8) : 1}

    answer_02 = [1,1,1,0,0,0,-1,-1,-1]

    def testGraphCut02_A(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "A")
    def testGraphCut02_L(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "L")
    def testGraphCut02_d(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "d")
    def testGraphCut02_tl(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "tl")
    def testGraphCut02_ta(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "ta")
    def testGraphCut02_tfa(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "tfa")

    graph_03 = {
        (0,1) : 4,
        (0,2) : 3,
        (1,2) : 1,
        (1,3) : 1,
        (1,3) : 1,
        (2,2) : 4,
        (2,4) : 3,
        (3,5) : 2,
        (4,5) : 5,
        (5,5) : 10,
        (5,6) : 0,   # not really there
        (6,6) : 1,
        (6,7) : 2,
        (7,8) : 1}

    answer_03 = [1,1,1,0,0,0,-1,-1,-1]

    def testGraphCut02_A(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "A")
    def testGraphCut02_L(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "L")
    def testGraphCut02_d(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "d")
    def testGraphCut02_tl(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "tl")
    def testGraphCut02_ta(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "ta")
    def testGraphCut02_tfa(self): self.checkCut(self.graph_02, 0, 5, self.answer_02, "tfa")

    ############################################################
    # Now testing for bad things happening with the graph cuts

    def testGraphCutBad_ArrayNonSquare(self):
        self.assertRaises(ValueError, lambda: graphCut(array([[0,1], [1,0], [0,0]]), 0,1)) 

    def testGraphCutBad_ListNonSquare(self):
        self.assertRaises(ValueError, lambda: graphCut([[0,1], [1,0], [0,0]], 0,1))

    def testGraphCutBad_ArrayNegCapacity(self):
        self.assertRaises(ValueError, lambda: graphCut(array([[0,1], [1,0], [0,0]]), 0,1)) 

    graph_bad_01 = {
        (0,1) : 4,
        (0,2) : 3,
        (1,2) : 1,
        (1,3) : 1,
        (1,3) : 1,
        (2,2) : 4,
        (2,4) : 3,
        (3,5) : -2,
        (4,5) : 5,
        (5,5) : 10,

        (6,6) : 1,
        (6,7) : 2,
        (7,8) : 1}

    answer_bad_01 = [1,1,1,0,0,0,-1,-1,-1]

    def testGraphCut_bad_01_A(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "A"))
    def testGraphCut_bad_bad_01_L(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "L"))
    def testGraphCut_bad_bad_01_d(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "d"))
    def testGraphCut_bad_bad_01_tl(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "tl"))
    def testGraphCut_bad_bad_01_ta(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "ta"))
    def testGraphCut_bad_bad_01_tfa(self): 
        self.assertRaises(ValueError, lambda:self.checkCut(self.graph_bad_01, 0, 5, self.answer_bad_01, "tfa"))


class TestPotentialMaximizing(unittest.TestCase):

    def checkPFunc(self, E1, E2, answer, opts):
        
        s_edges = [i for i,j in E2.iterkeys()]
        d_edges = [j for i,j in E2.iterkeys()]
        E2v      = E2.values()
        
        nV = max(max(s_edges), max(d_edges)) + 1

        def makeArray():
            A = zeros( (nV, nV, 4), dtype=float64)

            for (i, j), c in E2.iteritems():
                A[i,j, :] = array(c)

            return A
                

        E1d = {}
        
        E1d["l"] = list(E1)
        E1d["a"] = array(E1)

        E2d = {}

        E2d["l"] = (s_edges, d_edges, E2v)
        E2d["a"] = (array(s_edges, uint), array(d_edges,uint), array(E2v))
        E2d["L"] = (s_edges, d_edges, array(E2v))
        E2d["f"] = (array(s_edges, float64), array(d_edges, float64), array(E2v))
        E2d["A"] = makeArray()
        E2d["S"] = makeArray() + makeArray().transpose(1,0,2)
        E2d["d"] = E2

        ret = maximimzeGraphPotential(E1d[opts[0]], E2d[opts[1]])

        self.assert_(len(ret) == len(answer))

        for re, ae in zip(ret, answer):
            self.assert_(re == ae, "%s != %s (true)" % (str(ret), str(answer)))

    ising_01_E2 = {
        (0,1) : (10,0,0,10),
        (1,2) : (5,0,0,5),
        (2,3) : (1,0,0,1),   # should break here
        (3,4) : (5,0,0,5),
        (4,5) : (10,0,0,10)}

    ising_01_E1 = [10, 0, 0, 0, 0, -10] 
        
    ising_01_answer = [1,1,1,0,0,0]


    def testSimpleIsing_01_ll(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "ll")
    def testSimpleIsing_01_la(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "la")
    def testSimpleIsing_01_lL(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "lL")
    def testSimpleIsing_01_lf(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "lf")
    def testSimpleIsing_01_lA(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "lA")
    def testSimpleIsing_01_lS(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "lS")
    def testSimpleIsing_01_ld(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "ld")

    def testSimpleIsing_01_al(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "al")
    def testSimpleIsing_01_aa(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "aa")
    def testSimpleIsing_01_aL(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "aL")
    def testSimpleIsing_01_af(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "af")
    def testSimpleIsing_01_aA(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "aA")
    def testSimpleIsing_01_aS(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "aS")
    def testSimpleIsing_01_ad(self):
        self.checkPFunc(self.ising_01_E1, self.ising_01_E2, self.ising_01_answer, "ad")


    ising_02_E2 = {
        (1,0) : (10,0,0,10),
        (2,1) : (5,0,0,5),
        (3,2) : (1,0,0,1),   # should break here
        (4,3) : (5,0,0,5),
        (5,4) : (10,0,0,10)}

    ising_02_E1 = [10, 0, 0, 0, 0, -10] 
        
    ising_02_answer = [1,1,1,0,0,0]

    def testSimpleIsing_02_ll(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "ll")
    def testSimpleIsing_02_la(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "la")
    def testSimpleIsing_02_lL(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "lL")
    def testSimpleIsing_02_lf(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "lf")
    def testSimpleIsing_02_lA(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "lA")
    def testSimpleIsing_02_lS(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "lS")
    def testSimpleIsing_02_ld(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "ld")

    def testSimpleIsing_02_al(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "al")
    def testSimpleIsing_02_aa(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "aa")
    def testSimpleIsing_02_aL(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "aL")
    def testSimpleIsing_02_af(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "af")
    def testSimpleIsing_02_aA(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "aA")
    def testSimpleIsing_02_aS(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "aS")
    def testSimpleIsing_02_ad(self):
        self.checkPFunc(self.ising_02_E1, self.ising_02_E2, self.ising_02_answer, "ad")


    ising_03_E2 = {
        (1,0) : (10,0,0,10),
        (2,1) : (0,0,0,0),   # 
        (4,3) : (0,0,0,0),   #
        (5,4) : (10,0,0,10)}

    ising_03_E1 = [1e-8, 0, 0, 0, 0, -1e-8]
        
    ising_03_answer = [1,1,-1,-1,0,0]

    def testSimpleIsing_03_ll(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "ll")
    def testSimpleIsing_03_la(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "la")
    def testSimpleIsing_03_lL(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "lL")
    def testSimpleIsing_03_lf(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "lf")
    def testSimpleIsing_03_lA(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "lA")
    def testSimpleIsing_03_lS(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "lS")
    def testSimpleIsing_03_ld(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "ld")

    def testSimpleIsing_03_al(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "al")
    def testSimpleIsing_03_aa(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "aa")
    def testSimpleIsing_03_aL(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "aL")
    def testSimpleIsing_03_af(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "af")
    def testSimpleIsing_03_aA(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "aA")
    def testSimpleIsing_03_aS(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "aS")
    def testSimpleIsing_03_ad(self):
        self.checkPFunc(self.ising_03_E1, self.ising_03_E2, self.ising_03_answer, "ad")

        

if __name__ == '__main__':
    unittest.main()

