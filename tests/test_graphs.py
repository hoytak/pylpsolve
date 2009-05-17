import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array, ones, eye, float64, uint, zeros

from pylpsolve.graphs import graphCut

class TestGraphs(unittest.TestCase):

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
    # Now testing for bad things happening

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

if __name__ == '__main__':
    unittest.main()

