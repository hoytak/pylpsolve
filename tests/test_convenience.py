import random, unittest, cPickle, collections
from copy import deepcopy, copy
from pylpsolve import LP, LPException
from numpy import array as ar, ones, eye, float64, uint


############################################################
# Tests the convenience functions

class TestConvenience(unittest.TestCase):

    def checkBindEach(self, opts):

        idxlist = [{}, {}]
        
        idxlist[0]["t"] = (0,3)
        idxlist[0]["N"] = "a"
        idxlist[0]["l"] = [0,1,2]
        idxlist[0]["a"] = ar([0,1,2])
        idxlist[0]["r"] = ar([0,0,1,1,2,2])[::2]
        idxlist[0]["f"] = ar([0,1,2],dtype=float64)

        idxlist[1]["t"] = (3,6)
        idxlist[1]["n"] = "b"
        idxlist[1]["l"] = [3,4,5]
        idxlist[1]["a"] = ar([3,4,5])
        idxlist[1]["r"] = ar([3,3,4,4,5,5])[::2]
        idxlist[1]["f"] = ar([3,4,5],dtype=float64)

        lp = LP()

        if opts[0] == "N":
            self.assert_(lp.getVariables(idxlist[0]["N"], 3) == (0,3) )

        # Now bind the second group
        if opts[2] == "g":
            self.assert_(
                lp.bindEach(idxlist[1][opts[1]], ">", idxlist[0][opts[0]])
                == [0,1,2])

        elif opts[2] == "l":
            self.assert_(
                lp.bindEach(idxlist[0][opts[0]], "<", idxlist[1][opts[1]])
                == [0,1,2])

        elif opts[2] == "e":
            self.assert_(
                lp.bindEach(idxlist[0][opts[0]], "=", idxlist[1][opts[1]])
                == [0,1,2])
        elif opts[2] == "E":
            self.assert_(
                lp.bindEach(idxlist[1][opts[1]], "=", idxlist[0][opts[0]])
                == [0,1,2])
        else:
            assert False

        # Forces some to be defined implicitly above to catch that case
        lp.addConstraint( (idxlist[0][opts[0]], 1), ">=", 1)

        lp.setObjective( (idxlist[1][opts[1]], [1,2,3]) )
        lp.setMinimize()

        lp.solve()

        v = lp.getSolution()

        self.assert_(len(v) == 6, "len(v) = %d != 6" % len(v))
        self.assertAlmostEqual(v[0], 1)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(v[2], 0)
        self.assertAlmostEqual(v[3], 1)
        self.assertAlmostEqual(v[4], 0)
        self.assertAlmostEqual(v[5], 0)

        if opts[0] in "nN" and opts[1] in "nN":

            d = lp.getSolutionDict()

            self.assert_(set(d.iterkeys()) == set(["a", "b"]))

            self.assertAlmostEqual(d["a"][0], 1)
            self.assertAlmostEqual(d["a"][1], 0)
            self.assertAlmostEqual(d["a"][2], 0)
            self.assertAlmostEqual(d["b"][0], 1)
            self.assertAlmostEqual(d["b"][1], 0)
            self.assertAlmostEqual(d["b"][2], 0)



    def testBindEach_ttg(self): self.checkBindEach("ttg")
    def testBindEach_ttl(self): self.checkBindEach("ttl")
    def testBindEach_tte(self): self.checkBindEach("tte")
    def testBindEach_ttE(self): self.checkBindEach("ttE")

    def testBindEach_tng(self): self.checkBindEach("tng")
    def testBindEach_tnl(self): self.checkBindEach("tnl")
    def testBindEach_tne(self): self.checkBindEach("tne")
    def testBindEach_tnE(self): self.checkBindEach("tnE")

    def testBindEach_tlg(self): self.checkBindEach("tlg")
    def testBindEach_tll(self): self.checkBindEach("tll")
    def testBindEach_tle(self): self.checkBindEach("tle")
    def testBindEach_tlE(self): self.checkBindEach("tlE")

    def testBindEach_tag(self): self.checkBindEach("tag")
    def testBindEach_tal(self): self.checkBindEach("tal")
    def testBindEach_tae(self): self.checkBindEach("tae")
    def testBindEach_taE(self): self.checkBindEach("taE")

    def testBindEach_trg(self): self.checkBindEach("trg")
    def testBindEach_trl(self): self.checkBindEach("trl")
    def testBindEach_tre(self): self.checkBindEach("tre")
    def testBindEach_trE(self): self.checkBindEach("trE")

    def testBindEach_tfg(self): self.checkBindEach("tfg")
    def testBindEach_tfl(self): self.checkBindEach("tfl")
    def testBindEach_tfe(self): self.checkBindEach("tfe")
    def testBindEach_tfE(self): self.checkBindEach("tfE")


    def testBindEach_Ntg(self): self.checkBindEach("Ntg")
    def testBindEach_Ntl(self): self.checkBindEach("Ntl")
    def testBindEach_Nte(self): self.checkBindEach("Nte")
    def testBindEach_NtE(self): self.checkBindEach("NtE")

    def testBindEach_Nng(self): self.checkBindEach("Nng")
    def testBindEach_Nnl(self): self.checkBindEach("Nnl")
    def testBindEach_Nne(self): self.checkBindEach("Nne")
    def testBindEach_NnE(self): self.checkBindEach("NnE")

    def testBindEach_Nlg(self): self.checkBindEach("Nlg")
    def testBindEach_Nll(self): self.checkBindEach("Nll")
    def testBindEach_Nle(self): self.checkBindEach("Nle")
    def testBindEach_NlE(self): self.checkBindEach("NlE")

    def testBindEach_Nag(self): self.checkBindEach("Nag")
    def testBindEach_Nal(self): self.checkBindEach("Nal")
    def testBindEach_Nae(self): self.checkBindEach("Nae")
    def testBindEach_NaE(self): self.checkBindEach("NaE")

    def testBindEach_Nrg(self): self.checkBindEach("Nrg")
    def testBindEach_Nrl(self): self.checkBindEach("Nrl")
    def testBindEach_Nre(self): self.checkBindEach("Nre")
    def testBindEach_NrE(self): self.checkBindEach("NrE")

    def testBindEach_Nfg(self): self.checkBindEach("Nfg")
    def testBindEach_Nfl(self): self.checkBindEach("Nfl")
    def testBindEach_Nfe(self): self.checkBindEach("Nfe")
    def testBindEach_NfE(self): self.checkBindEach("NfE")


    def testBindEach_ltg(self): self.checkBindEach("ltg")
    def testBindEach_ltl(self): self.checkBindEach("ltl")
    def testBindEach_lte(self): self.checkBindEach("lte")
    def testBindEach_ltE(self): self.checkBindEach("ltE")

    def testBindEach_lng(self): self.checkBindEach("lng")
    def testBindEach_lnl(self): self.checkBindEach("lnl")
    def testBindEach_lne(self): self.checkBindEach("lne")
    def testBindEach_lnE(self): self.checkBindEach("lnE")

    def testBindEach_llg(self): self.checkBindEach("llg")
    def testBindEach_lll(self): self.checkBindEach("lll")
    def testBindEach_lle(self): self.checkBindEach("lle")
    def testBindEach_llE(self): self.checkBindEach("llE")

    def testBindEach_lag(self): self.checkBindEach("lag")
    def testBindEach_lal(self): self.checkBindEach("lal")
    def testBindEach_lae(self): self.checkBindEach("lae")
    def testBindEach_laE(self): self.checkBindEach("laE")

    def testBindEach_lrg(self): self.checkBindEach("lrg")
    def testBindEach_lrl(self): self.checkBindEach("lrl")
    def testBindEach_lre(self): self.checkBindEach("lre")
    def testBindEach_lrE(self): self.checkBindEach("lrE")

    def testBindEach_lfg(self): self.checkBindEach("lfg")
    def testBindEach_lfl(self): self.checkBindEach("lfl")
    def testBindEach_lfe(self): self.checkBindEach("lfe")
    def testBindEach_lfE(self): self.checkBindEach("lfE")


    def testBindEach_atg(self): self.checkBindEach("atg")
    def testBindEach_atl(self): self.checkBindEach("atl")
    def testBindEach_ate(self): self.checkBindEach("ate")
    def testBindEach_atE(self): self.checkBindEach("atE")

    def testBindEach_ang(self): self.checkBindEach("ang")
    def testBindEach_anl(self): self.checkBindEach("anl")
    def testBindEach_ane(self): self.checkBindEach("ane")
    def testBindEach_anE(self): self.checkBindEach("anE")

    def testBindEach_alg(self): self.checkBindEach("alg")
    def testBindEach_all(self): self.checkBindEach("all")
    def testBindEach_ale(self): self.checkBindEach("ale")
    def testBindEach_alE(self): self.checkBindEach("alE")

    def testBindEach_aag(self): self.checkBindEach("aag")
    def testBindEach_aal(self): self.checkBindEach("aal")
    def testBindEach_aae(self): self.checkBindEach("aae")
    def testBindEach_aaE(self): self.checkBindEach("aaE")

    def testBindEach_arg(self): self.checkBindEach("arg")
    def testBindEach_arl(self): self.checkBindEach("arl")
    def testBindEach_are(self): self.checkBindEach("are")
    def testBindEach_arE(self): self.checkBindEach("arE")

    def testBindEach_afg(self): self.checkBindEach("afg")
    def testBindEach_afl(self): self.checkBindEach("afl")
    def testBindEach_afe(self): self.checkBindEach("afe")
    def testBindEach_afE(self): self.checkBindEach("afE")


    def testBindEach_rtg(self): self.checkBindEach("rtg")
    def testBindEach_rtl(self): self.checkBindEach("rtl")
    def testBindEach_rte(self): self.checkBindEach("rte")
    def testBindEach_rtE(self): self.checkBindEach("rtE")

    def testBindEach_rng(self): self.checkBindEach("rng")
    def testBindEach_rnl(self): self.checkBindEach("rnl")
    def testBindEach_rne(self): self.checkBindEach("rne")
    def testBindEach_rnE(self): self.checkBindEach("rnE")

    def testBindEach_rlg(self): self.checkBindEach("rlg")
    def testBindEach_rll(self): self.checkBindEach("rll")
    def testBindEach_rle(self): self.checkBindEach("rle")
    def testBindEach_rlE(self): self.checkBindEach("rlE")

    def testBindEach_rag(self): self.checkBindEach("rag")
    def testBindEach_ral(self): self.checkBindEach("ral")
    def testBindEach_rae(self): self.checkBindEach("rae")
    def testBindEach_raE(self): self.checkBindEach("raE")

    def testBindEach_rrg(self): self.checkBindEach("rrg")
    def testBindEach_rrl(self): self.checkBindEach("rrl")
    def testBindEach_rre(self): self.checkBindEach("rre")
    def testBindEach_rrE(self): self.checkBindEach("rrE")

    def testBindEach_rfg(self): self.checkBindEach("rfg")
    def testBindEach_rfl(self): self.checkBindEach("rfl")
    def testBindEach_rfe(self): self.checkBindEach("rfe")
    def testBindEach_rfE(self): self.checkBindEach("rfE")


    def testBindEach_ftg(self): self.checkBindEach("ftg")
    def testBindEach_ftl(self): self.checkBindEach("ftl")
    def testBindEach_fte(self): self.checkBindEach("fte")
    def testBindEach_ftE(self): self.checkBindEach("ftE")

    def testBindEach_fng(self): self.checkBindEach("fng")
    def testBindEach_fnl(self): self.checkBindEach("fnl")
    def testBindEach_fne(self): self.checkBindEach("fne")
    def testBindEach_fnE(self): self.checkBindEach("fnE")

    def testBindEach_flg(self): self.checkBindEach("flg")
    def testBindEach_fll(self): self.checkBindEach("fll")
    def testBindEach_fle(self): self.checkBindEach("fle")
    def testBindEach_flE(self): self.checkBindEach("flE")

    def testBindEach_fag(self): self.checkBindEach("fag")
    def testBindEach_fal(self): self.checkBindEach("fal")
    def testBindEach_fae(self): self.checkBindEach("fae")
    def testBindEach_faE(self): self.checkBindEach("faE")

    def testBindEach_frg(self): self.checkBindEach("frg")
    def testBindEach_frl(self): self.checkBindEach("frl")
    def testBindEach_fre(self): self.checkBindEach("fre")
    def testBindEach_frE(self): self.checkBindEach("frE")

    def testBindEach_ffg(self): self.checkBindEach("ffg")
    def testBindEach_ffl(self): self.checkBindEach("ffl")
    def testBindEach_ffe(self): self.checkBindEach("ffe")
    def testBindEach_ffE(self): self.checkBindEach("ffE")


    def checkBindSandwich(self, opts):

        idxlist = [{}, {}]
        
        idxlist[0]["t"] = (0,3)
        idxlist[0]["N"] = "a"
        idxlist[0]["l"] = [0,1,2]
        idxlist[0]["a"] = ar([0,1,2])
        idxlist[0]["r"] = ar([0,0,1,1,2,2])[::2]
        idxlist[0]["f"] = ar([0,1,2],dtype=float64)

        idxlist[1]["t"] = (3,6)
        idxlist[1]["n"] = "b"
        idxlist[1]["l"] = [3,4,5]
        idxlist[1]["a"] = ar([3,4,5])
        idxlist[1]["r"] = ar([3,3,4,4,5,5])[::2]
        idxlist[1]["f"] = ar([3,4,5],dtype=float64)

        lp = LP()

        if opts[0] == "N":
            self.assert_(lp.getVariables(idxlist[0]["N"], 3) == (0,3) )

        # Now bind the second group
            
        lp.bindSandwich(idxlist[0][opts[0]], idxlist[1][opts[1]])

        if opts[2] == "u":
            lp.addConstraint( (idxlist[0][opts[0]], 1), ">=", 1)
        elif opts[2] == "l":
            lp.addConstraint( (idxlist[0][opts[0]], 1), "<=", -1)
            lp.setUnbounded(idxlist[0][opts[0]])
        else:
            assert False

        lp.setObjective( (idxlist[1][opts[1]], [1,2,3]) )
        lp.setMinimize()

        lp.solve()

        v = lp.getSolution()

        v0 = 1 if opts[2] == "u" else -1

        self.assert_(len(v) == 6, "len(v) = %d != 6" % len(v))
        self.assertAlmostEqual(v[0], v0)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(v[2], 0)
        self.assertAlmostEqual(v[3], 1)
        self.assertAlmostEqual(v[4], 0)
        self.assertAlmostEqual(v[5], 0)

        if opts[0] in "nN" and opts[1] in "nN":

            d = lp.getSolutionDict()

            self.assert_(set(d.iterkeys()) == set(["a", "b"]))

            self.assertAlmostEqual(d["a"][0], v0)
            self.assertAlmostEqual(d["a"][1], 0)
            self.assertAlmostEqual(d["a"][2], 0)
            self.assertAlmostEqual(d["b"][0], 1)
            self.assertAlmostEqual(d["b"][1], 0)
            self.assertAlmostEqual(d["b"][2], 0)

    def testBindSandwich_ttu(self): self.checkBindSandwich("ttu")
    def testBindSandwich_ttl(self): self.checkBindSandwich("ttl")

    def testBindSandwich_tnu(self): self.checkBindSandwich("tnu")
    def testBindSandwich_tnl(self): self.checkBindSandwich("tnl")

    def testBindSandwich_tlu(self): self.checkBindSandwich("tlu")
    def testBindSandwich_tll(self): self.checkBindSandwich("tll")

    def testBindSandwich_tau(self): self.checkBindSandwich("tau")
    def testBindSandwich_tal(self): self.checkBindSandwich("tal")

    def testBindSandwich_tru(self): self.checkBindSandwich("tru")
    def testBindSandwich_trl(self): self.checkBindSandwich("trl")

    def testBindSandwich_tfu(self): self.checkBindSandwich("tfu")
    def testBindSandwich_tfl(self): self.checkBindSandwich("tfl")


    def testBindSandwich_Ntu(self): self.checkBindSandwich("Ntu")
    def testBindSandwich_Ntl(self): self.checkBindSandwich("Ntl")

    def testBindSandwich_Nnu(self): self.checkBindSandwich("Nnu")
    def testBindSandwich_Nnl(self): self.checkBindSandwich("Nnl")

    def testBindSandwich_Nlu(self): self.checkBindSandwich("Nlu")
    def testBindSandwich_Nll(self): self.checkBindSandwich("Nll")

    def testBindSandwich_Nau(self): self.checkBindSandwich("Nau")
    def testBindSandwich_Nal(self): self.checkBindSandwich("Nal")

    def testBindSandwich_Nru(self): self.checkBindSandwich("Nru")
    def testBindSandwich_Nrl(self): self.checkBindSandwich("Nrl")

    def testBindSandwich_Nfu(self): self.checkBindSandwich("Nfu")
    def testBindSandwich_Nfl(self): self.checkBindSandwich("Nfl")


    def testBindSandwich_ltu(self): self.checkBindSandwich("ltu")
    def testBindSandwich_ltl(self): self.checkBindSandwich("ltl")

    def testBindSandwich_lnu(self): self.checkBindSandwich("lnu")
    def testBindSandwich_lnl(self): self.checkBindSandwich("lnl")

    def testBindSandwich_llu(self): self.checkBindSandwich("llu")
    def testBindSandwich_lll(self): self.checkBindSandwich("lll")

    def testBindSandwich_lau(self): self.checkBindSandwich("lau")
    def testBindSandwich_lal(self): self.checkBindSandwich("lal")

    def testBindSandwich_lru(self): self.checkBindSandwich("lru")
    def testBindSandwich_lrl(self): self.checkBindSandwich("lrl")

    def testBindSandwich_lfu(self): self.checkBindSandwich("lfu")
    def testBindSandwich_lfl(self): self.checkBindSandwich("lfl")


    def testBindSandwich_atu(self): self.checkBindSandwich("atu")
    def testBindSandwich_atl(self): self.checkBindSandwich("atl")

    def testBindSandwich_anu(self): self.checkBindSandwich("anu")
    def testBindSandwich_anl(self): self.checkBindSandwich("anl")

    def testBindSandwich_alu(self): self.checkBindSandwich("alu")
    def testBindSandwich_all(self): self.checkBindSandwich("all")

    def testBindSandwich_aau(self): self.checkBindSandwich("aau")
    def testBindSandwich_aal(self): self.checkBindSandwich("aal")

    def testBindSandwich_aru(self): self.checkBindSandwich("aru")
    def testBindSandwich_arl(self): self.checkBindSandwich("arl")

    def testBindSandwich_afu(self): self.checkBindSandwich("afu")
    def testBindSandwich_afl(self): self.checkBindSandwich("afl")


    def testBindSandwich_rtu(self): self.checkBindSandwich("rtu")
    def testBindSandwich_rtl(self): self.checkBindSandwich("rtl")

    def testBindSandwich_rnu(self): self.checkBindSandwich("rnu")
    def testBindSandwich_rnl(self): self.checkBindSandwich("rnl")

    def testBindSandwich_rlu(self): self.checkBindSandwich("rlu")
    def testBindSandwich_rll(self): self.checkBindSandwich("rll")

    def testBindSandwich_rau(self): self.checkBindSandwich("rau")
    def testBindSandwich_ral(self): self.checkBindSandwich("ral")

    def testBindSandwich_rru(self): self.checkBindSandwich("rru")
    def testBindSandwich_rrl(self): self.checkBindSandwich("rrl")

    def testBindSandwich_rfu(self): self.checkBindSandwich("rfu")
    def testBindSandwich_rfl(self): self.checkBindSandwich("rfl")


    def testBindSandwich_ftu(self): self.checkBindSandwich("ftu")
    def testBindSandwich_ftl(self): self.checkBindSandwich("ftl")

    def testBindSandwich_fnu(self): self.checkBindSandwich("fnu")
    def testBindSandwich_fnl(self): self.checkBindSandwich("fnl")

    def testBindSandwich_flu(self): self.checkBindSandwich("flu")
    def testBindSandwich_fll(self): self.checkBindSandwich("fll")

    def testBindSandwich_fau(self): self.checkBindSandwich("fau")
    def testBindSandwich_fal(self): self.checkBindSandwich("fal")

    def testBindSandwich_fru(self): self.checkBindSandwich("fru")
    def testBindSandwich_frl(self): self.checkBindSandwich("frl")

    def testBindSandwich_ffu(self): self.checkBindSandwich("ffu")
    def testBindSandwich_ffl(self): self.checkBindSandwich("ffl")
