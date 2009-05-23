import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LPSolve, LPSolveException
from numpy import array as ar, ones, eye, float64, uint

# TODO:

# test that running with maximize, then minimize works with no
# other changes.

class TestBasic(unittest.TestCase):
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
        
        lp.addConstraint( (ar([0]), ar([1])) , ">", 1)
        lp.setObjective( [1], mode = "minimize")

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

    def test02_basic_minimize(self):
        lp = LPSolve()

        self.assert_(lp.addConstraint(1, ">", 10) == 0)

        lp.setObjective(1)

        # Minimize
        lp.setMinimize()
        lp.solve()
        self.assertAlmostEqual(lp.getObjectiveValue(), 10)


    def test03_basic_maximize(self):
        lp = LPSolve()

        self.assert_(lp.addConstraint(1, "<", 20) == 0)

        lp.setObjective(1)

        # Maximize
        lp.setMaximize()
        lp.solve()
        self.assertAlmostEqual(lp.getObjectiveValue(), 20)

    def test04_basic_maxmin_rerun(self):
        lp = LPSolve()

        self.assert_(lp.addConstraint(1, ">", 10) == 0)
        self.assert_(lp.addConstraint(1, "<", 20) == 1)

        lp.setObjective(1)

        # Minimize
        lp.setMinimize()
        lp.solve()
        self.assertAlmostEqual(lp.getObjectiveValue(), 10)

        # Maximize
        lp.setMaximize()
        lp.solve()
        self.assertAlmostEqual(lp.getObjectiveValue(), 20)


class TestMinimal(unittest.TestCase):

    def checkMinLP1(self, opts):
        
        lp = LPSolve()

        indices = {}
        indices["t"] = (0,3)
        indices["n"] = "a"
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2])
        indices["r"] = ar([2,1,0])[::-1]
        indices["f"] = ar([0,1,2],dtype=float64)
        indices["e"] = None  # empty

        weights = {}
        weights["l"] = [1,1,1]
        weights["a"] = ar([1,1,1])
        weights["f"] = ar([1,1,1])
        weights["r"] = ar([1,1,1])[::-1]
        weights["s"] = 1

        obj_func = {}
        obj_func["l"] = [1,2,3]
        obj_func["a"] = ar([1,2,3])
        obj_func["f"] = ar([1,2,3],dtype=float64)
        obj_func["r"] = ar([3,2,1],dtype=float64)[::-1]
        obj_func["s"] = [1,2,3]  # can't do scalar here


        # Some ones used in the dict's case
        il = indices["l"] 
        assert len(il) == 3

        wl = weights["l"]
        assert len(wl) == 3

        ol = obj_func["l"]
        assert len(ol) == 3

        if opts[0] == "d" or opts[0] == "T":

            if opts[1] == "1":
                cd = [ (i, w) for i, w in zip(il, wl)]
                od = [ (i, o) for i, o in zip(il, ol)]

            elif opts[1] == "2":
                cd = [ ("a", wl[:2]), ("b", wl[2])]
                od = [ ("a", ol[:2]), ("b", ol[2])]
            
            elif opts[1] == "3":
                cd = [((0,2), wl[:2]), (2, wl[2])]
                od = [((0,2), ol[:2]), (2, ol[2])]

            elif opts[1] == "4":
                cd = [((0,2), wl[:2]), ( (2,3), wl[2])]
                od = [((0,2), ol[:2]), ( (2,3), ol[2])]

            elif opts[1] == "5":  # bad for out of order
                cd = [("a", wl[:2]), ( (2,3), wl[2])]
                od = [("a", ol[:2]), ( (2,3), ol[2])]
            
            elif opts[1] in indices.keys() and opts[2] in weights.keys():
                cd = [(indices[opts[1]], weights[opts[2]])]
                od = [(indices[opts[1]], obj_func[opts[2]])]
            else:
                assert False

            if opts[0] == "d":
                self.assert_(lp.addConstraint(dict(cd), ">=", 1) == 0)
                lp.setObjective(dict(od))
            elif opts[0] == "T":

                if opts[1] == "N":
                    lp.getVariables(indices["N"], 3)

                self.assert_(lp.addConstraint(cd, ">=", 1) == 0)
                lp.setObjective(od)
        else:
            assert len(opts) == 2
            
            if opts[0] == "N":
                lp.getVariables(indices["N"], 3)

            io = indices[opts[0]]

            if io is None:
                self.assert_(lp.addConstraint(weights[opts[1]], ">=", 1) == 0)
                lp.setObjective(obj_func[opts[1]])
            else:
                self.assert_(lp.addConstraint( (indices[opts[0]], weights[opts[1]]), ">=", 1) == 0)
                lp.setObjective( (indices[opts[0]], obj_func[opts[1]]))

        lp.solve()

        self.assertAlmostEqual(lp.getObjectiveValue(), 1)
        
        if opts[0] not in ["d", "T"]:
            v = lp.getSolution(indices[opts[0]])
        else:
            v = lp.getSolution()

        self.assert_(len(v) == 3, "len(v) = %d != 3" % len(v))

        self.assertAlmostEqual(lp.getSolution(0), 1)
        self.assertAlmostEqual(v[0], 1)
        self.assertAlmostEqual(lp.getSolution(1), 0)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(lp.getSolution(2), 0)
        self.assertAlmostEqual(v[2], 0)


    def testConstraints_tl(self): self.checkMinLP1("tl")
    def testConstraints_ta(self): self.checkMinLP1("ta")
    def testConstraints_tf(self): self.checkMinLP1("tf")
    def testConstraints_ts(self): self.checkMinLP1("ts")
    def testConstraints_tr(self): self.checkMinLP1("tr")

    def testConstraints_nl(self): self.checkMinLP1("nl")
    def testConstraints_na(self): self.checkMinLP1("na")
    def testConstraints_nf(self): self.checkMinLP1("nf")
    def testConstraints_nr(self): self.checkMinLP1("nr")

    def testConstraints_Nl(self): self.checkMinLP1("Nl")
    def testConstraints_Na(self): self.checkMinLP1("Na")
    def testConstraints_Nf(self): self.checkMinLP1("Nf")
    def testConstraints_Ns(self): self.checkMinLP1("Ns")
    def testConstraints_Nr(self): self.checkMinLP1("Nr")

    def testConstraints_ll(self): self.checkMinLP1("ll")
    def testConstraints_la(self): self.checkMinLP1("la")
    def testConstraints_lf(self): self.checkMinLP1("lf")
    def testConstraints_ls(self): self.checkMinLP1("ls")
    def testConstraints_lr(self): self.checkMinLP1("lr")

    def testConstraints_al(self): self.checkMinLP1("al")
    def testConstraints_aa(self): self.checkMinLP1("aa")
    def testConstraints_af(self): self.checkMinLP1("af")
    def testConstraints_as(self): self.checkMinLP1("as")
    def testConstraints_ar(self): self.checkMinLP1("ar")

    def testConstraints_fl(self): self.checkMinLP1("fl")
    def testConstraints_fa(self): self.checkMinLP1("fa")
    def testConstraints_ff(self): self.checkMinLP1("ff")
    def testConstraints_fs(self): self.checkMinLP1("fs")
    def testConstraints_fr(self): self.checkMinLP1("fr")

    def testConstraints_el(self): self.checkMinLP1("el")
    def testConstraints_ea(self): self.checkMinLP1("ea")
    def testConstraints_ef(self): self.checkMinLP1("ef")
    def testConstraints_er(self): self.checkMinLP1("er")

    def testConstraints_rl(self): self.checkMinLP1("rl")
    def testConstraints_ra(self): self.checkMinLP1("ra")
    def testConstraints_rf(self): self.checkMinLP1("rf")
    def testConstraints_rs(self): self.checkMinLP1("rs")
    def testConstraints_rr(self): self.checkMinLP1("rr")


    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")

    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")

    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")

    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")

    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")

    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")

    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")

    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")

    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")


    def testConstraints_Ttl(self): self.checkMinLP1("Ttl")
    def testConstraints_Tta(self): self.checkMinLP1("Tta")
    def testConstraints_Ttf(self): self.checkMinLP1("Ttf")
    def testConstraints_Tts(self): self.checkMinLP1("Tts")
    def testConstraints_Ttr(self): self.checkMinLP1("Ttr")

    def testConstraints_Tnl(self): self.checkMinLP1("Tnl")
    def testConstraints_Tna(self): self.checkMinLP1("Tna")
    def testConstraints_Tnf(self): self.checkMinLP1("Tnf")
    def testConstraints_Tnr(self): self.checkMinLP1("Tnr")

    def testConstraints_TNl(self): self.checkMinLP1("TNl")
    def testConstraints_TNa(self): self.checkMinLP1("TNa")
    def testConstraints_TNf(self): self.checkMinLP1("TNf")
    def testConstraints_TNs(self): self.checkMinLP1("TNs")
    def testConstraints_TNr(self): self.checkMinLP1("TNr")

    def testConstraints_Tll(self): self.checkMinLP1("Tll")
    def testConstraints_Tla(self): self.checkMinLP1("Tla")
    def testConstraints_Tlf(self): self.checkMinLP1("Tlf")
    def testConstraints_Tls(self): self.checkMinLP1("Tls")
    def testConstraints_Tlr(self): self.checkMinLP1("Tlr")

    def testConstraints_Tal(self): self.checkMinLP1("Tal")
    def testConstraints_Taa(self): self.checkMinLP1("Taa")
    def testConstraints_Taf(self): self.checkMinLP1("Taf")
    def testConstraints_Tas(self): self.checkMinLP1("Tas")
    def testConstraints_Tar(self): self.checkMinLP1("Tar")

    def testConstraints_Tfl(self): self.checkMinLP1("Tfl")
    def testConstraints_Tfa(self): self.checkMinLP1("Tfa")
    def testConstraints_Tff(self): self.checkMinLP1("Tff")
    def testConstraints_Tfs(self): self.checkMinLP1("Tfs")
    def testConstraints_Tfr(self): self.checkMinLP1("Tfr")

    def testConstraints_Tel(self): self.checkMinLP1("Tel")
    def testConstraints_Tea(self): self.checkMinLP1("Tea")
    def testConstraints_Tef(self): self.checkMinLP1("Tef")
    def testConstraints_Ter(self): self.checkMinLP1("Ter")

    def testConstraints_Trl(self): self.checkMinLP1("Trl")
    def testConstraints_Tra(self): self.checkMinLP1("Tra")
    def testConstraints_Trf(self): self.checkMinLP1("Trf")
    def testConstraints_Trs(self): self.checkMinLP1("Trs")
    def testConstraints_Trr(self): self.checkMinLP1("Trr")



class TestTwoLevel(unittest.TestCase):

    def checkMinLP1(self, opts):
        
        lp = LPSolve()

        idxlist = [{}, {}]

        idxlist[0]["t"] = (0,3)
        idxlist[0]["n"] = "a"
        idxlist[0]["N"] = "a"
        idxlist[0]["l"] = [0,1,2]
        idxlist[0]["a"] = ar([0,1,2])
        idxlist[0]["r"] = ar([0,0,1,1,2,2])[::2]
        idxlist[0]["f"] = ar([0,1,2],dtype=float64)
        idxlist[0]["e"] = None  # empty

        idxlist[1]["t"] = (3,6)
        idxlist[1]["n"] = "b"
        idxlist[1]["N"] = "b"
        idxlist[1]["l"] = [3,4,5]
        idxlist[1]["a"] = ar([3,4,5])
        idxlist[1]["r"] = ar([3,3,4,4,5,5])[::2]
        idxlist[1]["f"] = ar([3,4,5],dtype=float64)
        idxlist[1]["e"] = (3,6)

        weightlist = [{}, {}]
        weightlist[0]["l"] = [1,1,1]
        weightlist[0]["a"] = ar([1,1,1])
        weightlist[0]["f"] = ar([1,1,1])
        weightlist[0]["r"] = ar([1,1,1,1,1,1])[::2]
        weightlist[0]["s"] = 1

        weightlist[1]["l"] = [1,0.5,0.5]
        weightlist[1]["a"] = ar([1,0.5,0.5])
        weightlist[1]["f"] = ar([1,0.5,0.5])
        weightlist[1]["r"] = ar([1,1,0.5,0.5,0.5,0.5])[::2]
        weightlist[1]["s"] = [1.0, 0.5, 0.5]

        obj_func_list = [{},{}]
        obj_func_list[0]["l"] = [1,2,3]
        obj_func_list[0]["a"] = ar([1,2,3])
        obj_func_list[0]["f"] = ar([1,2,3],dtype=float64)
        obj_func_list[0]["r"] = ar([1,1,2,2,3,3],dtype=float64)[::2]
        obj_func_list[0]["s"] = [1,2,3]  # can't do scalar here

        obj_func_list[1]["l"] = [1,1,1]
        obj_func_list[1]["a"] = ar([1,1,1])
        obj_func_list[1]["f"] = ar([1,1,1],dtype=float64)
        obj_func_list[1]["r"] = ar([1,1,1,1,1,1],dtype=float64)[::2]
        obj_func_list[1]["s"] = 1

        register_check = {}
        disable_regular_check = False

        for ci, (indices, weights, obj_func) in enumerate(zip(idxlist, weightlist, obj_func_list)):

            # Some ones used in the dict's case
            il = indices["l"]
            assert len(il) == 3

            wl = weights["l"]
            assert len(wl) == 3

            ol = obj_func["l"]
            assert len(ol) == 3

            if opts[0] == "d" or opts[0] == "T":

                t = il[0]
                assert il[-1] - il[0] == 2

                n1 = indices["n"]
                n2 = indices["n"]+"2"

                if opts[1] == "1":
                    cd = [ (i, w) for i, w in zip(il, wl)]
                    od = [ (i, o) for i, o in zip(il, ol)]

                elif opts[1] == "2":
                    cd = [ (n1, wl[:2]), (n2, wl[2])]
                    od = [ (n1, ol[:2]), (n2, ol[2])]
                    
                    register_check[n1] = [1,0]
                    register_check[n2] = [0]
                    disable_regular_check = True

                elif opts[1] == "3":
                    cd = [((t,t+2), wl[:2]), (t+2, wl[2])]
                    od = [((t,t+2), ol[:2]), (t+2, wl[2])]

                elif opts[1] == "4":
                    cd = [((t,t+2), wl[:2]), ((t+2,t+3), wl[2])]
                    od = [((t,t+2), ol[:2]), ((t+2,t+3), wl[2])]

                elif opts[1] == "5":  # bad for dict
                    cd = [(n1, wl[:2]), ((t+2,t+3), wl[2])]
                    od = [(n1, ol[:2]), ((t+2,t+3), wl[2])]

                elif opts[1] in indices.keys() and opts[2] in weights.keys():
                    cd = [(indices[opts[1]], weights[opts[2]])]
                    od = [(indices[opts[1]], obj_func[opts[2]])]

                else:
                    assert False

                if opts[0] == "d":
                    self.assert_(lp.addConstraint(dict(cd), ">=", 1) == ci)
                    lp.addToObjective(dict(od))
                elif opts[0] == "T":

                    if opts[1] == "N":
                        lp.getVariables(indices["N"], 3)

                    self.assert_(lp.addConstraint(cd, ">=", 1) == ci)
                    lp.addToObjective(od)
            else:
                assert len(opts) == 2

                if opts[0] == "N":
                    lp.getVariables(indices["N"], 3)

                io = indices[opts[0]]

                if io is None:
                    self.assert_(lp.addConstraint(weights[opts[1]], ">=", 1) == ci)
                    lp.addToObjective(obj_func[opts[1]])
                else:
                    self.assert_(lp.addConstraint( (indices[opts[0]], weights[opts[1]]), ">=", 1) == ci)
                    lp.addToObjective( (indices[opts[0]], obj_func[opts[1]]))

        for num_times in range(2):
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), 2)

            if disable_regular_check:
                for k, l in register_check.iteritems():
                    v = lp.getSolution(k)
                    self.assert_(len(v) == len(l))
                    for i1,i2 in zip(l,v):
                        self.assertAlmostEqual(i1,i2)
            else:
                v = lp.getSolution()

                self.assert_(len(v) == 6, "len(v) = %d != 6" % len(v))
                self.assertAlmostEqual(v[0], 1)
                self.assertAlmostEqual(v[1], 0)
                self.assertAlmostEqual(v[2], 0)
                self.assertAlmostEqual(v[3], 1)
                self.assertAlmostEqual(v[4], 0)
                self.assertAlmostEqual(v[5], 0)

                if opts[0] in "nN":

                    d = lp.getSolutionDict()

                    self.assert_(set(d.iterkeys()) == set(["a", "b"]))

                    self.assertAlmostEqual(d["a"][0], 1)
                    self.assertAlmostEqual(d["a"][1], 0)
                    self.assertAlmostEqual(d["a"][2], 0)
                    self.assertAlmostEqual(d["b"][0], 1)
                    self.assertAlmostEqual(d["b"][1], 0)
                    self.assertAlmostEqual(d["b"][2], 0)

            # now test the retrieval

    def testConstraints_tl(self): self.checkMinLP1("tl")
    def testConstraints_ta(self): self.checkMinLP1("ta")
    def testConstraints_tf(self): self.checkMinLP1("tf")
    def testConstraints_ts(self): self.checkMinLP1("ts")
    def testConstraints_tr(self): self.checkMinLP1("tr")

    def testConstraints_nl(self): self.checkMinLP1("nl")
    def testConstraints_na(self): self.checkMinLP1("na")
    def testConstraints_nf(self): self.checkMinLP1("nf")
    def testConstraints_nr(self): self.checkMinLP1("nr")

    def testConstraints_Nl(self): self.checkMinLP1("Nl")
    def testConstraints_Na(self): self.checkMinLP1("Na")
    def testConstraints_Nf(self): self.checkMinLP1("Nf")
    def testConstraints_Ns(self): self.checkMinLP1("Ns")
    def testConstraints_Nr(self): self.checkMinLP1("Nr")

    def testConstraints_ll(self): self.checkMinLP1("ll")
    def testConstraints_la(self): self.checkMinLP1("la")
    def testConstraints_lf(self): self.checkMinLP1("lf")
    def testConstraints_ls(self): self.checkMinLP1("ls")
    def testConstraints_lr(self): self.checkMinLP1("lr")

    def testConstraints_al(self): self.checkMinLP1("al")
    def testConstraints_aa(self): self.checkMinLP1("aa")
    def testConstraints_af(self): self.checkMinLP1("af")
    def testConstraints_as(self): self.checkMinLP1("as")
    def testConstraints_ar(self): self.checkMinLP1("ar")

    def testConstraints_fl(self): self.checkMinLP1("fl")
    def testConstraints_fa(self): self.checkMinLP1("fa")
    def testConstraints_ff(self): self.checkMinLP1("ff")
    def testConstraints_fs(self): self.checkMinLP1("fs")
    def testConstraints_fr(self): self.checkMinLP1("fr")

    def testConstraints_el(self): self.checkMinLP1("el")
    def testConstraints_ea(self): self.checkMinLP1("ea")
    def testConstraints_ef(self): self.checkMinLP1("ef")
    def testConstraints_er(self): self.checkMinLP1("er")

    def testConstraints_rl(self): self.checkMinLP1("rl")
    def testConstraints_ra(self): self.checkMinLP1("ra")
    def testConstraints_rf(self): self.checkMinLP1("rf")
    def testConstraints_rs(self): self.checkMinLP1("rs")
    def testConstraints_rr(self): self.checkMinLP1("rr")


    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")

    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")

    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")

    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")

    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")

    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")

    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")

    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")

    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")


    def testConstraints_Ttl(self): self.checkMinLP1("Ttl")
    def testConstraints_Tta(self): self.checkMinLP1("Tta")
    def testConstraints_Ttf(self): self.checkMinLP1("Ttf")
    def testConstraints_Tts(self): self.checkMinLP1("Tts")
    def testConstraints_Ttr(self): self.checkMinLP1("Ttr")

    def testConstraints_Tnl(self): self.checkMinLP1("Tnl")
    def testConstraints_Tna(self): self.checkMinLP1("Tna")
    def testConstraints_Tnf(self): self.checkMinLP1("Tnf")
    def testConstraints_Tnr(self): self.checkMinLP1("Tnr")

    def testConstraints_TNl(self): self.checkMinLP1("TNl")
    def testConstraints_TNa(self): self.checkMinLP1("TNa")
    def testConstraints_TNf(self): self.checkMinLP1("TNf")
    def testConstraints_TNs(self): self.checkMinLP1("TNs")
    def testConstraints_TNr(self): self.checkMinLP1("TNr")

    def testConstraints_Tll(self): self.checkMinLP1("Tll")
    def testConstraints_Tla(self): self.checkMinLP1("Tla")
    def testConstraints_Tlf(self): self.checkMinLP1("Tlf")
    def testConstraints_Tls(self): self.checkMinLP1("Tls")
    def testConstraints_Tlr(self): self.checkMinLP1("Tlr")

    def testConstraints_Tal(self): self.checkMinLP1("Tal")
    def testConstraints_Taa(self): self.checkMinLP1("Taa")
    def testConstraints_Taf(self): self.checkMinLP1("Taf")
    def testConstraints_Tas(self): self.checkMinLP1("Tas")
    def testConstraints_Tar(self): self.checkMinLP1("Tar")

    def testConstraints_Tfl(self): self.checkMinLP1("Tfl")
    def testConstraints_Tfa(self): self.checkMinLP1("Tfa")
    def testConstraints_Tff(self): self.checkMinLP1("Tff")
    def testConstraints_Tfs(self): self.checkMinLP1("Tfs")
    def testConstraints_Tfr(self): self.checkMinLP1("Tfr")

    def testConstraints_Tel(self): self.checkMinLP1("Tel")
    def testConstraints_Tea(self): self.checkMinLP1("Tea")
    def testConstraints_Tef(self): self.checkMinLP1("Tef")
    def testConstraints_Ter(self): self.checkMinLP1("Ter")

    def testConstraints_Trl(self): self.checkMinLP1("Trl")
    def testConstraints_Tra(self): self.checkMinLP1("Tra")
    def testConstraints_Trf(self): self.checkMinLP1("Trf")
    def testConstraints_Trs(self): self.checkMinLP1("Trs")
    def testConstraints_Trr(self): self.checkMinLP1("Trr")
            


    ############################################################
    # Test 2d stuff

    def check2dMatrix(self, opts):

        values = {}

        indices = {}
        indices["t"] = (0,3)
        indices["n"] = "a"
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2],dtype=uint)
        indices["r"] = ar([0,0,1,1,2,2],dtype=uint)[::2]
        indices["f"] = ar([0,1,2],dtype=float64)
        indices["e"] = None  # empty

        A = [[1,0,  0],
             [0,0,  0.5],
             [0,0.5,0]]

        values = {}
        values["L"] = A
        values["l"] = [ar(le) for le in A]
        values["a"] = ar(A)

        targets = {}
        targets["s"] = 1
        targets["l"] = [1,1,1]
        targets["a"] = ar([1,1,1],dtype=uint)
        targets["r"] = ar([1,1,1,1,1,1],dtype=uint)[::2]
        targets["f"] = ar([1,1,1],dtype=float64)

        lp = LPSolve()

        if opts[0] == "N":
            lp.getVariables(indices["N"], 3)

        io = indices[opts[0]]
        vl = values [opts[1]]
        tr = targets[opts[2]]
        ob = [1,2,3]
        
        c_ret_idx = [0,1,2]

        if io is None:
            ret = lp.addConstraint(vl, ">=", tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective(ob)
        else:
            ret = lp.addConstraint( (io, vl), ">=", tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective( (io, ob) )

        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), 11)

            v = lp.getSolution()

            self.assert_(len(v) == 3)
            self.assertAlmostEqual(v[0], 1)
            self.assertAlmostEqual(v[1], 2)
            self.assertAlmostEqual(v[2], 2)


    def test2DMatrix_tLs(self): self.check2dMatrix("tLs")
    def test2DMatrix_tLl(self): self.check2dMatrix("tLl")
    def test2DMatrix_tLa(self): self.check2dMatrix("tLa")
    def test2DMatrix_tLf(self): self.check2dMatrix("tLf")
    def test2DMatrix_tLr(self): self.check2dMatrix("tLr")

    def test2DMatrix_tls(self): self.check2dMatrix("tls")
    def test2DMatrix_tll(self): self.check2dMatrix("tll")
    def test2DMatrix_tla(self): self.check2dMatrix("tla")
    def test2DMatrix_tlf(self): self.check2dMatrix("tlf")
    def test2DMatrix_tlr(self): self.check2dMatrix("tlr")

    def test2DMatrix_tas(self): self.check2dMatrix("tls")
    def test2DMatrix_tal(self): self.check2dMatrix("tll")
    def test2DMatrix_taa(self): self.check2dMatrix("tla")
    def test2DMatrix_taf(self): self.check2dMatrix("tlf")
    def test2DMatrix_tar(self): self.check2dMatrix("tlr")

    
    def test2DMatrix_nLs(self): self.check2dMatrix("nLs")
    def test2DMatrix_nLl(self): self.check2dMatrix("nLl")
    def test2DMatrix_nLa(self): self.check2dMatrix("nLa")
    def test2DMatrix_nLf(self): self.check2dMatrix("nLf")
    def test2DMatrix_nLr(self): self.check2dMatrix("nLr")

    def test2DMatrix_nls(self): self.check2dMatrix("nls")
    def test2DMatrix_nll(self): self.check2dMatrix("nll")
    def test2DMatrix_nla(self): self.check2dMatrix("nla")
    def test2DMatrix_nlf(self): self.check2dMatrix("nlf")
    def test2DMatrix_nlr(self): self.check2dMatrix("nlr")

    def test2DMatrix_nas(self): self.check2dMatrix("nls")
    def test2DMatrix_nal(self): self.check2dMatrix("nll")
    def test2DMatrix_naa(self): self.check2dMatrix("nla")
    def test2DMatrix_naf(self): self.check2dMatrix("nlf")


    def test2DMatrix_NLs(self): self.check2dMatrix("NLs")
    def test2DMatrix_NLl(self): self.check2dMatrix("NLl")
    def test2DMatrix_NLa(self): self.check2dMatrix("NLa")
    def test2DMatrix_NLf(self): self.check2dMatrix("NLf")
    def test2DMatrix_NLr(self): self.check2dMatrix("NLr")

    def test2DMatrix_Nls(self): self.check2dMatrix("Nls")
    def test2DMatrix_Nll(self): self.check2dMatrix("Nll")
    def test2DMatrix_Nla(self): self.check2dMatrix("Nla")
    def test2DMatrix_Nlf(self): self.check2dMatrix("Nlf")
    def test2DMatrix_Nlr(self): self.check2dMatrix("Nlr")

    def test2DMatrix_Nas(self): self.check2dMatrix("Nls")
    def test2DMatrix_Nal(self): self.check2dMatrix("Nll")
    def test2DMatrix_Naa(self): self.check2dMatrix("Nla")
    def test2DMatrix_Naf(self): self.check2dMatrix("Nlf")
    def test2DMatrix_Nar(self): self.check2dMatrix("Nlr")


    def test2DMatrix_lLs(self): self.check2dMatrix("lLs")
    def test2DMatrix_lLl(self): self.check2dMatrix("lLl")
    def test2DMatrix_lLa(self): self.check2dMatrix("lLa")
    def test2DMatrix_lLf(self): self.check2dMatrix("lLf")
    def test2DMatrix_lLr(self): self.check2dMatrix("lLr")

    def test2DMatrix_lls(self): self.check2dMatrix("lls")
    def test2DMatrix_lll(self): self.check2dMatrix("lll")
    def test2DMatrix_lla(self): self.check2dMatrix("lla")
    def test2DMatrix_llf(self): self.check2dMatrix("llf")
    def test2DMatrix_llr(self): self.check2dMatrix("llr")

    def test2DMatrix_las(self): self.check2dMatrix("lls")
    def test2DMatrix_lal(self): self.check2dMatrix("lll")
    def test2DMatrix_laa(self): self.check2dMatrix("lla")
    def test2DMatrix_laf(self): self.check2dMatrix("llf")
    def test2DMatrix_lar(self): self.check2dMatrix("llr")


    def test2DMatrix_aLs(self): self.check2dMatrix("aLs")
    def test2DMatrix_aLl(self): self.check2dMatrix("aLl")
    def test2DMatrix_aLa(self): self.check2dMatrix("aLa")
    def test2DMatrix_aLf(self): self.check2dMatrix("aLf")
    def test2DMatrix_aLr(self): self.check2dMatrix("aLr")

    def test2DMatrix_als(self): self.check2dMatrix("als")
    def test2DMatrix_all(self): self.check2dMatrix("all")
    def test2DMatrix_ala(self): self.check2dMatrix("ala")
    def test2DMatrix_alf(self): self.check2dMatrix("alf")
    def test2DMatrix_alr(self): self.check2dMatrix("alr")

    def test2DMatrix_aas(self): self.check2dMatrix("als")
    def test2DMatrix_aal(self): self.check2dMatrix("all")
    def test2DMatrix_aaa(self): self.check2dMatrix("ala")
    def test2DMatrix_aaf(self): self.check2dMatrix("alf")
    def test2DMatrix_aar(self): self.check2dMatrix("alr")


    def test2DMatrix_fLs(self): self.check2dMatrix("fLs")
    def test2DMatrix_fLl(self): self.check2dMatrix("fLl")
    def test2DMatrix_fLa(self): self.check2dMatrix("fLa")
    def test2DMatrix_fLf(self): self.check2dMatrix("fLf")
    def test2DMatrix_fLr(self): self.check2dMatrix("fLr")

    def test2DMatrix_fls(self): self.check2dMatrix("fls")
    def test2DMatrix_fll(self): self.check2dMatrix("fll")
    def test2DMatrix_fla(self): self.check2dMatrix("fla")
    def test2DMatrix_flf(self): self.check2dMatrix("flf")
    def test2DMatrix_flr(self): self.check2dMatrix("flr")

    def test2DMatrix_fas(self): self.check2dMatrix("fls")
    def test2DMatrix_fal(self): self.check2dMatrix("fll")
    def test2DMatrix_faa(self): self.check2dMatrix("fla")
    def test2DMatrix_faf(self): self.check2dMatrix("flf")
    def test2DMatrix_far(self): self.check2dMatrix("flr")


    def test2DMatrix_eLs(self): self.check2dMatrix("eLs")
    def test2DMatrix_eLl(self): self.check2dMatrix("eLl")
    def test2DMatrix_eLa(self): self.check2dMatrix("eLa")
    def test2DMatrix_eLf(self): self.check2dMatrix("eLf")
    def test2DMatrix_eLr(self): self.check2dMatrix("eLr")

    def test2DMatrix_els(self): self.check2dMatrix("els")
    def test2DMatrix_ell(self): self.check2dMatrix("ell")
    def test2DMatrix_ela(self): self.check2dMatrix("ela")
    def test2DMatrix_elf(self): self.check2dMatrix("elf")
    def test2DMatrix_elr(self): self.check2dMatrix("elr")

    def test2DMatrix_eas(self): self.check2dMatrix("els")
    def test2DMatrix_eal(self): self.check2dMatrix("ell")
    def test2DMatrix_eaa(self): self.check2dMatrix("ela")
    def test2DMatrix_eaf(self): self.check2dMatrix("elf")
    def test2DMatrix_ear(self): self.check2dMatrix("elr")


    def test2DMatrix_rLs(self): self.check2dMatrix("rLs")
    def test2DMatrix_rLl(self): self.check2dMatrix("rLl")
    def test2DMatrix_rLa(self): self.check2dMatrix("rLa")
    def test2DMatrix_rLf(self): self.check2dMatrix("rLf")
    def test2DMatrix_rLr(self): self.check2dMatrix("rLr")

    def test2DMatrix_rls(self): self.check2dMatrix("rls")
    def test2DMatrix_rll(self): self.check2dMatrix("rll")
    def test2DMatrix_rla(self): self.check2dMatrix("rla")
    def test2DMatrix_rlf(self): self.check2dMatrix("rlf")
    def test2DMatrix_rlr(self): self.check2dMatrix("rlr")

    def test2DMatrix_ras(self): self.check2dMatrix("rls")
    def test2DMatrix_ral(self): self.check2dMatrix("rll")
    def test2DMatrix_raa(self): self.check2dMatrix("rla")
    def test2DMatrix_raf(self): self.check2dMatrix("rlf")
    def test2DMatrix_rar(self): self.check2dMatrix("rlr")


if __name__ == '__main__':
    unittest.main()

