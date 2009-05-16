import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array as ar, ones, eye, float64, uint, int

class TestErrorCatch(unittest.TestCase):
    # test constraint adding by (wrong typed index array, value array)
    def test01_constraint_rejects_float_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.addConstraint( (ar([0, 1.1, 2],dtype=float64), ar([1,1,1],dtype=float64) ), ">=", 1))

    # test constraint adding by (wrong typed index array, value array)
    def test01_objfunc_rejects_float_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.setObjective( (ar([0, 1.1, 2],dtype=float64), ar([1,1,1],dtype=float64) )))

    def test02_constraint_rejects_neg_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.addConstraint( (ar([0, -1, 2]), ar([1,1,1],dtype=float64) ), ">=", 1))

    def test02_objfunc_rejects_negative_idx(self):
        lp = LPSolve()

        self.assertRaises(ValueError,
                          lambda: lp.setObjective( (ar([0, -1, 2]), ar([1,1,1],dtype=float64) )))

    
    def checkBadSizingTooLarge(self, opts):
        
        lp = LPSolve()

        def run_test(c_arg, o_arg):
            if opts[-1] == "c":
                self.assertRaises(ValueError, lambda: lp.addConstraint(c_arg, ">", 1))
            elif opts[-1] == "o":
                self.assertRaises(ValueError, lambda: lp.setObjective(o_arg))
            else:
                assert False

        indices = {}
        indices["t"] = (0,3)
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2])
        indices["f"] = ar([0,1,2],dtype=float64)

        weights = {}
        weights["l"] = [1,1,1,1]
        weights["a"] = ar([1,1,1,1])
        weights["f"] = ar([1,1,1,1])

        obj_func = {}
        obj_func["l"] = [1,2,3,4]
        obj_func["a"] = ar([1,2,3,4])
        obj_func["f"] = ar([1,2,3,4],dtype=float64)


        # Some ones used in the dict's case
        il = indices["l"] 
        assert len(il) == 3

        wl = weights["l"]
        assert len(wl) == 4

        ol = obj_func["l"]
        assert len(ol) == 4

        if opts[0] == "d" or opts[0] == "T":

            if opts[1] == "2":
                lp.getVariables("b", 1)
                cd = [ ("a", wl[:2]), ("b", wl[2:])]
                od = [ ("a", ol[:2]), ("b", ol[2:])]
            
            elif opts[1] == "3":
                cd = [((0,2), wl[:2]), (2, wl[2:])]
                od = [((0,2), ol[:2]), (2, ol[2:])]

            elif opts[1] == "4":
                cd = [((0,2), wl[:2]), ( (2,3), wl[2:])]
                od = [((0,2), ol[:2]), ( (2,3), ol[2:])]

            elif opts[1] == "5":  # bad for out of order
                cd = [("a", wl[:2]), ( (2,3), wl[2:])]
                od = [("a", ol[:2]), ( (2,3), ol[2:])]
            
            elif opts[1] in indices.keys() and opts[2] in weights.keys():

                if "N" in opts:
                    lp.getVariables(indices["N"], 3)

                cd = [(indices[opts[1]], weights[opts[2]])]
                od = [(indices[opts[1]], obj_func[opts[2]])]
            else:
                assert False

            if opts[0] == "d":
                run_test(dict(cd), dict(od))
                return
            elif opts[0] == "T":
                run_test(cd, od)
                return
            else:
                assert False
        else:
            assert len(opts) == 3
            
            # No little n option here
            if "N" in opts:
                lp.getVariables(indices["N"], 3)
            
            run_test( (indices[opts[0]], weights[opts[1]]),
                      (indices[opts[0]], obj_func[opts[1]]))
            return

    def testBadSizingTooLarge_tlc(self): self.checkBadSizingTooLarge("tlc")
    def testBadSizingTooLarge_tac(self): self.checkBadSizingTooLarge("tac")
    def testBadSizingTooLarge_tfc(self): self.checkBadSizingTooLarge("tfc")

    def testBadSizingTooLarge_Nlc(self): self.checkBadSizingTooLarge("Nlc")
    def testBadSizingTooLarge_Nac(self): self.checkBadSizingTooLarge("Nac")
    def testBadSizingTooLarge_Nfc(self): self.checkBadSizingTooLarge("Nfc")

    def testBadSizingTooLarge_llc(self): self.checkBadSizingTooLarge("llc")
    def testBadSizingTooLarge_lac(self): self.checkBadSizingTooLarge("lac")
    def testBadSizingTooLarge_lfc(self): self.checkBadSizingTooLarge("lfc")

    def testBadSizingTooLarge_alc(self): self.checkBadSizingTooLarge("alc")
    def testBadSizingTooLarge_aac(self): self.checkBadSizingTooLarge("aac")
    def testBadSizingTooLarge_afc(self): self.checkBadSizingTooLarge("afc")

    def testBadSizingTooLarge_flc(self): self.checkBadSizingTooLarge("flc")
    def testBadSizingTooLarge_fac(self): self.checkBadSizingTooLarge("fac")
    def testBadSizingTooLarge_ffc(self): self.checkBadSizingTooLarge("ffc")

    def testBadSizingTooLarge_d2c(self): self.checkBadSizingTooLarge("d2c")
    def testBadSizingTooLarge_d2c(self): self.checkBadSizingTooLarge("d2c")
    def testBadSizingTooLarge_d2c(self): self.checkBadSizingTooLarge("d2c")

    def testBadSizingTooLarge_d3c(self): self.checkBadSizingTooLarge("d3c")
    def testBadSizingTooLarge_d3c(self): self.checkBadSizingTooLarge("d3c")
    def testBadSizingTooLarge_d3c(self): self.checkBadSizingTooLarge("d3c")

    def testBadSizingTooLarge_d4c(self): self.checkBadSizingTooLarge("d4c")
    def testBadSizingTooLarge_d4c(self): self.checkBadSizingTooLarge("d4c")
    def testBadSizingTooLarge_d4c(self): self.checkBadSizingTooLarge("d4c")

    def testBadSizingTooLarge_T2c(self): self.checkBadSizingTooLarge("T2c")
    def testBadSizingTooLarge_T2c(self): self.checkBadSizingTooLarge("T2c")
    def testBadSizingTooLarge_T2c(self): self.checkBadSizingTooLarge("T2c")

    def testBadSizingTooLarge_T3c(self): self.checkBadSizingTooLarge("T3c")
    def testBadSizingTooLarge_T3c(self): self.checkBadSizingTooLarge("T3c")
    def testBadSizingTooLarge_T3c(self): self.checkBadSizingTooLarge("T3c")

    def testBadSizingTooLarge_T4c(self): self.checkBadSizingTooLarge("T4c")
    def testBadSizingTooLarge_T4c(self): self.checkBadSizingTooLarge("T4c")
    def testBadSizingTooLarge_T4c(self): self.checkBadSizingTooLarge("T4c")

    def testBadSizingTooLarge_T5c(self): self.checkBadSizingTooLarge("T5c")
    def testBadSizingTooLarge_T5c(self): self.checkBadSizingTooLarge("T5c")
    def testBadSizingTooLarge_T5c(self): self.checkBadSizingTooLarge("T5c")


    def testBadSizingTooLarge_Ttlo(self): self.checkBadSizingTooLarge("Ttlc")
    def testBadSizingTooLarge_Ttao(self): self.checkBadSizingTooLarge("Ttac")
    def testBadSizingTooLarge_Ttfo(self): self.checkBadSizingTooLarge("Ttfc")

    def testBadSizingTooLarge_TNlo(self): self.checkBadSizingTooLarge("TNlc")
    def testBadSizingTooLarge_TNao(self): self.checkBadSizingTooLarge("TNac")
    def testBadSizingTooLarge_TNfo(self): self.checkBadSizingTooLarge("TNfc")

    def testBadSizingTooLarge_Tllo(self): self.checkBadSizingTooLarge("Tllc")
    def testBadSizingTooLarge_Tlao(self): self.checkBadSizingTooLarge("Tlac")
    def testBadSizingTooLarge_Tlfo(self): self.checkBadSizingTooLarge("Tlfc")

    def testBadSizingTooLarge_Talo(self): self.checkBadSizingTooLarge("Talc")
    def testBadSizingTooLarge_Taao(self): self.checkBadSizingTooLarge("Taac")
    def testBadSizingTooLarge_Tafo(self): self.checkBadSizingTooLarge("Tafc")

    def testBadSizingTooLarge_Tflo(self): self.checkBadSizingTooLarge("Tflc")
    def testBadSizingTooLarge_Tfao(self): self.checkBadSizingTooLarge("Tfac")
    def testBadSizingTooLarge_Tffo(self): self.checkBadSizingTooLarge("Tffc")


    def testBadSizingTooLarge_tlo(self): self.checkBadSizingTooLarge("tlc")
    def testBadSizingTooLarge_tao(self): self.checkBadSizingTooLarge("tac")
    def testBadSizingTooLarge_tfo(self): self.checkBadSizingTooLarge("tfc")

    def testBadSizingTooLarge_Nlo(self): self.checkBadSizingTooLarge("Nlc")
    def testBadSizingTooLarge_Nao(self): self.checkBadSizingTooLarge("Nac")
    def testBadSizingTooLarge_Nfo(self): self.checkBadSizingTooLarge("Nfc")

    def testBadSizingTooLarge_llo(self): self.checkBadSizingTooLarge("llc")
    def testBadSizingTooLarge_lao(self): self.checkBadSizingTooLarge("lac")
    def testBadSizingTooLarge_lfo(self): self.checkBadSizingTooLarge("lfc")

    def testBadSizingTooLarge_alo(self): self.checkBadSizingTooLarge("alc")
    def testBadSizingTooLarge_aao(self): self.checkBadSizingTooLarge("aac")
    def testBadSizingTooLarge_afo(self): self.checkBadSizingTooLarge("afc")

    def testBadSizingTooLarge_flo(self): self.checkBadSizingTooLarge("flc")
    def testBadSizingTooLarge_fao(self): self.checkBadSizingTooLarge("fac")
    def testBadSizingTooLarge_ffo(self): self.checkBadSizingTooLarge("ffc")

    def testBadSizingTooLarge_d2o(self): self.checkBadSizingTooLarge("d2c")
    def testBadSizingTooLarge_d2o(self): self.checkBadSizingTooLarge("d2c")
    def testBadSizingTooLarge_d2o(self): self.checkBadSizingTooLarge("d2c")

    def testBadSizingTooLarge_d3o(self): self.checkBadSizingTooLarge("d3c")
    def testBadSizingTooLarge_d3o(self): self.checkBadSizingTooLarge("d3c")
    def testBadSizingTooLarge_d3o(self): self.checkBadSizingTooLarge("d3c")

    def testBadSizingTooLarge_d4o(self): self.checkBadSizingTooLarge("d4c")
    def testBadSizingTooLarge_d4o(self): self.checkBadSizingTooLarge("d4c")
    def testBadSizingTooLarge_d4o(self): self.checkBadSizingTooLarge("d4c")

    def testBadSizingTooLarge_T2o(self): self.checkBadSizingTooLarge("T2c")
    def testBadSizingTooLarge_T2o(self): self.checkBadSizingTooLarge("T2c")
    def testBadSizingTooLarge_T2o(self): self.checkBadSizingTooLarge("T2c")

    def testBadSizingTooLarge_T3o(self): self.checkBadSizingTooLarge("T3c")
    def testBadSizingTooLarge_T3o(self): self.checkBadSizingTooLarge("T3c")
    def testBadSizingTooLarge_T3o(self): self.checkBadSizingTooLarge("T3c")

    def testBadSizingTooLarge_T4o(self): self.checkBadSizingTooLarge("T4c")
    def testBadSizingTooLarge_T4o(self): self.checkBadSizingTooLarge("T4c")
    def testBadSizingTooLarge_T4o(self): self.checkBadSizingTooLarge("T4c")

    def testBadSizingTooLarge_T5o(self): self.checkBadSizingTooLarge("T5c")
    def testBadSizingTooLarge_T5o(self): self.checkBadSizingTooLarge("T5c")
    def testBadSizingTooLarge_T5o(self): self.checkBadSizingTooLarge("T5c")


    def testBadSizingTooLarge_Ttlo(self): self.checkBadSizingTooLarge("Ttlo")
    def testBadSizingTooLarge_Ttao(self): self.checkBadSizingTooLarge("Ttao")
    def testBadSizingTooLarge_Ttfo(self): self.checkBadSizingTooLarge("Ttfo")

    def testBadSizingTooLarge_TNlo(self): self.checkBadSizingTooLarge("TNlo")
    def testBadSizingTooLarge_TNao(self): self.checkBadSizingTooLarge("TNao")
    def testBadSizingTooLarge_TNfo(self): self.checkBadSizingTooLarge("TNfo")

    def testBadSizingTooLarge_Tllo(self): self.checkBadSizingTooLarge("Tllo")
    def testBadSizingTooLarge_Tlao(self): self.checkBadSizingTooLarge("Tlao")
    def testBadSizingTooLarge_Tlfo(self): self.checkBadSizingTooLarge("Tlfo")

    def testBadSizingTooLarge_Talo(self): self.checkBadSizingTooLarge("Talo")
    def testBadSizingTooLarge_Taao(self): self.checkBadSizingTooLarge("Taao")
    def testBadSizingTooLarge_Tafo(self): self.checkBadSizingTooLarge("Tafo")

    def testBadSizingTooLarge_Tflo(self): self.checkBadSizingTooLarge("Tflo")
    def testBadSizingTooLarge_Tfao(self): self.checkBadSizingTooLarge("Tfao")
    def testBadSizingTooLarge_Tffo(self): self.checkBadSizingTooLarge("Tffo")


    def checkBadSizingTooSmall(self, opts):
        
        lp = LPSolve()

        def run_test(c_arg, o_arg):
            if opts[-1] == "c":
                self.assertRaises(ValueError, lambda: lp.addConstraint(c_arg, ">", 1))
            elif opts[-1] == "o":
                self.assertRaises(ValueError, lambda: lp.setObjective(o_arg))
            else:
                assert False

        indices = {}
        indices["t"] = (0,5)
        indices["N"] = "a"
        indices["l"] = [0,1,2,3,4]
        indices["a"] = ar([0,1,2,3,4])
        indices["f"] = ar([0,1,2,3,4],dtype=float64)

        weights = {}
        weights["l"] = [1,1,1,1]
        weights["a"] = ar([1,1,1,1])
        weights["f"] = ar([1,1,1,1])

        obj_func = {}
        obj_func["l"] = [1,2,3,4]
        obj_func["a"] = ar([1,2,3,4])
        obj_func["f"] = ar([1,2,3,4],dtype=float64)


        # Some ones used in the dict's case
        il = indices["l"] 
        assert len(il) == 5

        wl = weights["l"]
        assert len(wl) == 4

        ol = obj_func["l"]
        assert len(ol) == 4

        if opts[0] == "d" or opts[0] == "T":

            if opts[1] == "2":
                lp.getVariables("b", 3)
                cd = [ ("a", wl[:2]), ("b", wl[2:])]
                od = [ ("a", ol[:2]), ("b", ol[2:])]

            elif opts[1] == "4":
                cd = [((0,2), wl[:2]), ( (2,5), wl[2:])]
                od = [((0,2), ol[:2]), ( (2,5), ol[2:])]

            elif opts[1] == "5":  # bad for out of order
                cd = [("a", wl[:2]), ( (2,5), wl[2:])]
                od = [("a", ol[:2]), ( (2,5), ol[2:])]
            
            elif opts[1] in indices.keys() and opts[2] in weights.keys():

                if "N" in opts:
                    lp.getVariables(indices["N"], 5)

                cd = [(indices[opts[1]], weights[opts[2]])]
                od = [(indices[opts[1]], obj_func[opts[2]])]
            else:
                assert False

            if opts[0] == "d":
                run_test(dict(cd), dict(od))
                return
            elif opts[0] == "T":
                run_test(cd, od)
                return
            else:
                assert False
        else:
            assert len(opts) == 3
            
            # No little n option here
            if "N" in opts:
                lp.getVariables(indices["N"], 5)
            
            run_test( (indices[opts[0]], weights[opts[1]]),
                      (indices[opts[0]], obj_func[opts[1]]))
            return

    def testBadSizingTooSmall_tlc(self): self.checkBadSizingTooSmall("tlc")
    def testBadSizingTooSmall_tac(self): self.checkBadSizingTooSmall("tac")
    def testBadSizingTooSmall_tfc(self): self.checkBadSizingTooSmall("tfc")

    def testBadSizingTooSmall_Nlc(self): self.checkBadSizingTooSmall("Nlc")
    def testBadSizingTooSmall_Nac(self): self.checkBadSizingTooSmall("Nac")
    def testBadSizingTooSmall_Nfc(self): self.checkBadSizingTooSmall("Nfc")

    def testBadSizingTooSmall_llc(self): self.checkBadSizingTooSmall("llc")
    def testBadSizingTooSmall_lac(self): self.checkBadSizingTooSmall("lac")
    def testBadSizingTooSmall_lfc(self): self.checkBadSizingTooSmall("lfc")

    def testBadSizingTooSmall_alc(self): self.checkBadSizingTooSmall("alc")
    def testBadSizingTooSmall_aac(self): self.checkBadSizingTooSmall("aac")
    def testBadSizingTooSmall_afc(self): self.checkBadSizingTooSmall("afc")

    def testBadSizingTooSmall_flc(self): self.checkBadSizingTooSmall("flc")
    def testBadSizingTooSmall_fac(self): self.checkBadSizingTooSmall("fac")
    def testBadSizingTooSmall_ffc(self): self.checkBadSizingTooSmall("ffc")

    def testBadSizingTooSmall_d2c(self): self.checkBadSizingTooSmall("d2c")
    def testBadSizingTooSmall_d2c(self): self.checkBadSizingTooSmall("d2c")
    def testBadSizingTooSmall_d2c(self): self.checkBadSizingTooSmall("d2c")

    def testBadSizingTooSmall_d4c(self): self.checkBadSizingTooSmall("d4c")
    def testBadSizingTooSmall_d4c(self): self.checkBadSizingTooSmall("d4c")
    def testBadSizingTooSmall_d4c(self): self.checkBadSizingTooSmall("d4c")

    def testBadSizingTooSmall_T2c(self): self.checkBadSizingTooSmall("T2c")
    def testBadSizingTooSmall_T2c(self): self.checkBadSizingTooSmall("T2c")
    def testBadSizingTooSmall_T2c(self): self.checkBadSizingTooSmall("T2c")

    def testBadSizingTooSmall_T4c(self): self.checkBadSizingTooSmall("T4c")
    def testBadSizingTooSmall_T4c(self): self.checkBadSizingTooSmall("T4c")
    def testBadSizingTooSmall_T4c(self): self.checkBadSizingTooSmall("T4c")

    def testBadSizingTooSmall_T5c(self): self.checkBadSizingTooSmall("T5c")
    def testBadSizingTooSmall_T5c(self): self.checkBadSizingTooSmall("T5c")
    def testBadSizingTooSmall_T5c(self): self.checkBadSizingTooSmall("T5c")


    def testBadSizingTooSmall_Ttlo(self): self.checkBadSizingTooSmall("Ttlc")
    def testBadSizingTooSmall_Ttao(self): self.checkBadSizingTooSmall("Ttac")
    def testBadSizingTooSmall_Ttfo(self): self.checkBadSizingTooSmall("Ttfc")

    def testBadSizingTooSmall_TNlo(self): self.checkBadSizingTooSmall("TNlc")
    def testBadSizingTooSmall_TNao(self): self.checkBadSizingTooSmall("TNac")
    def testBadSizingTooSmall_TNfo(self): self.checkBadSizingTooSmall("TNfc")

    def testBadSizingTooSmall_Tllo(self): self.checkBadSizingTooSmall("Tllc")
    def testBadSizingTooSmall_Tlao(self): self.checkBadSizingTooSmall("Tlac")
    def testBadSizingTooSmall_Tlfo(self): self.checkBadSizingTooSmall("Tlfc")

    def testBadSizingTooSmall_Talo(self): self.checkBadSizingTooSmall("Talc")
    def testBadSizingTooSmall_Taao(self): self.checkBadSizingTooSmall("Taac")
    def testBadSizingTooSmall_Tafo(self): self.checkBadSizingTooSmall("Tafc")

    def testBadSizingTooSmall_Tflo(self): self.checkBadSizingTooSmall("Tflc")
    def testBadSizingTooSmall_Tfao(self): self.checkBadSizingTooSmall("Tfac")
    def testBadSizingTooSmall_Tffo(self): self.checkBadSizingTooSmall("Tffc")


    def testBadSizingTooSmall_tlo(self): self.checkBadSizingTooSmall("tlc")
    def testBadSizingTooSmall_tao(self): self.checkBadSizingTooSmall("tac")
    def testBadSizingTooSmall_tfo(self): self.checkBadSizingTooSmall("tfc")

    def testBadSizingTooSmall_Nlo(self): self.checkBadSizingTooSmall("Nlc")
    def testBadSizingTooSmall_Nao(self): self.checkBadSizingTooSmall("Nac")
    def testBadSizingTooSmall_Nfo(self): self.checkBadSizingTooSmall("Nfc")

    def testBadSizingTooSmall_llo(self): self.checkBadSizingTooSmall("llc")
    def testBadSizingTooSmall_lao(self): self.checkBadSizingTooSmall("lac")
    def testBadSizingTooSmall_lfo(self): self.checkBadSizingTooSmall("lfc")

    def testBadSizingTooSmall_alo(self): self.checkBadSizingTooSmall("alc")
    def testBadSizingTooSmall_aao(self): self.checkBadSizingTooSmall("aac")
    def testBadSizingTooSmall_afo(self): self.checkBadSizingTooSmall("afc")

    def testBadSizingTooSmall_flo(self): self.checkBadSizingTooSmall("flc")
    def testBadSizingTooSmall_fao(self): self.checkBadSizingTooSmall("fac")
    def testBadSizingTooSmall_ffo(self): self.checkBadSizingTooSmall("ffc")

    def testBadSizingTooSmall_d2o(self): self.checkBadSizingTooSmall("d2c")
    def testBadSizingTooSmall_d2o(self): self.checkBadSizingTooSmall("d2c")
    def testBadSizingTooSmall_d2o(self): self.checkBadSizingTooSmall("d2c")

    def testBadSizingTooSmall_d4o(self): self.checkBadSizingTooSmall("d4c")
    def testBadSizingTooSmall_d4o(self): self.checkBadSizingTooSmall("d4c")
    def testBadSizingTooSmall_d4o(self): self.checkBadSizingTooSmall("d4c")

    def testBadSizingTooSmall_T2o(self): self.checkBadSizingTooSmall("T2c")
    def testBadSizingTooSmall_T2o(self): self.checkBadSizingTooSmall("T2c")
    def testBadSizingTooSmall_T2o(self): self.checkBadSizingTooSmall("T2c")

    def testBadSizingTooSmall_T4o(self): self.checkBadSizingTooSmall("T4c")
    def testBadSizingTooSmall_T4o(self): self.checkBadSizingTooSmall("T4c")
    def testBadSizingTooSmall_T4o(self): self.checkBadSizingTooSmall("T4c")

    def testBadSizingTooSmall_T5o(self): self.checkBadSizingTooSmall("T5c")
    def testBadSizingTooSmall_T5o(self): self.checkBadSizingTooSmall("T5c")
    def testBadSizingTooSmall_T5o(self): self.checkBadSizingTooSmall("T5c")


    def testBadSizingTooSmall_Ttlo(self): self.checkBadSizingTooSmall("Ttlo")
    def testBadSizingTooSmall_Ttao(self): self.checkBadSizingTooSmall("Ttao")
    def testBadSizingTooSmall_Ttfo(self): self.checkBadSizingTooSmall("Ttfo")

    def testBadSizingTooSmall_TNlo(self): self.checkBadSizingTooSmall("TNlo")
    def testBadSizingTooSmall_TNao(self): self.checkBadSizingTooSmall("TNao")
    def testBadSizingTooSmall_TNfo(self): self.checkBadSizingTooSmall("TNfo")

    def testBadSizingTooSmall_Tllo(self): self.checkBadSizingTooSmall("Tllo")
    def testBadSizingTooSmall_Tlao(self): self.checkBadSizingTooSmall("Tlao")
    def testBadSizingTooSmall_Tlfo(self): self.checkBadSizingTooSmall("Tlfo")

    def testBadSizingTooSmall_Talo(self): self.checkBadSizingTooSmall("Talo")
    def testBadSizingTooSmall_Taao(self): self.checkBadSizingTooSmall("Taao")
    def testBadSizingTooSmall_Tafo(self): self.checkBadSizingTooSmall("Tafo")

    def testBadSizingTooSmall_Tflo(self): self.checkBadSizingTooSmall("Tflo")
    def testBadSizingTooSmall_Tfao(self): self.checkBadSizingTooSmall("Tfao")
    def testBadSizingTooSmall_Tffo(self): self.checkBadSizingTooSmall("Tffo")


if __name__ == '__main__':
    unittest.main()
