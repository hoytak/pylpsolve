import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LP, LPException
from numpy import ndarray as ar, ones, eye

class TestBlocks(unittest.TestCase):

    # Test retrieval of blocks
    def test01(self):
        lp = LP()

        self.assert_(lp.getIndexBlock(2, "a1") == (0,2))

    # Test retrieval of blocks
    def test01_r(self):
        lp = LP()

        self.assert_(lp.getIndexBlock("a1", 2) == (0,2))

    def test02(self):
        lp = LP()

        self.assert_(lp.getIndexBlock(2, "a1") == (0,2))
        self.assert_(lp.getIndexBlock(4, "a2") == (2,6))
        self.assert_(lp.getIndexBlock("a1") == (0,2))

    def test02_reverse_order(self):
        lp = LP()

        self.assert_(lp.getIndexBlock("a1", 2) == (0,2))
        self.assert_(lp.getIndexBlock("a2", 4) == (2,6))
        self.assert_(lp.getIndexBlock("a1", 2) == (0,2))

    def test02_reverse_order_mixed(self):
        lp = LP()

        self.assert_(lp.getIndexBlock("a1", 2) == (0,2))
        self.assert_(lp.getIndexBlock(4, "a2") == (2,6))
        self.assert_(lp.getIndexBlock("a1", 2) == (0,2))

    def test03_bad_recall(self):
        lp = LP()

        self.assert_(lp.getIndexBlock(2, "a1") == (0,2))
        self.assert_(lp.getIndexBlock(4, "a2") == (2,6))
        self.assertRaises(ValueError, lambda: lp.getIndexBlock(3, "a1"))

    def test04_bad_size_01(self):
        lp = LP()
        
        self.assertRaises(ValueError, lambda: lp.getIndexBlock("a1", 0))

    def test04_bad_size_02(self):
        lp = LP()
        
        self.assertRaises(ValueError, lambda: lp.getIndexBlock("a1", 0.5))

    def test04_bad_size_03(self):
        lp = LP()
        
        self.assertRaises(ValueError, lambda: lp.getIndexBlock("a1", -1))

    def test04_bad_size_04(self):
        lp = LP()
        
        self.assertRaises(ValueError, lambda: lp.getIndexBlock("a1", "a2"))

    def test04_bad_size_05(self):
        lp = LP()
        
        self.assertRaises(ValueError, lambda: lp.getIndexBlock(0, 2))

if __name__ == '__main__':
    unittest.main()

