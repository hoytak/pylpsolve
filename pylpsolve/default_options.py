# Sets the default options; this is also the authoritative list of
# what options are allowed and which aren't.

_default_options = {
    "presolve_none"        :   False,
    "presolve_rows"        :   False,
    "presolve_cols"        :   False,
    "presolve_lindep"      :   False,
    "presolve_aggregate"   :   False,
    "presolve_sparser"     :   True,
    "presolve_sos"         :   False,
    "presolve_reducemip"   :   False,
    "presolve_knapsack"    :   False,
    "presolve_elimeq2"     :   True,
    "presolve_impliedfree" :   False,
    "presolve_reducegcd"   :   False,
    "presolve_probefix"    :   False,
    "presolve_probereduce" :   False,
    "presolve_rowdominate" :   False,
    "presolve_coldominate" :   False,
    "presolve_mergerows"   :   False,
    "presolve_impliedslk"  :   False,
    "presolve_colfixdual"  :   False,
    "presolve_bounds"      :   False,
    "presolve_duals"       :   False,
    "presolve_sensduals"   :   False,

    # Pricing
    "pricer"               :  "devex",
    "price_primalfallback" :  False,
    "price_multiple"       :  False,
    "price_partial"        :  False,
    "price_adaptive"       :  True,
    "price_randomize"      :  False,
    "price_autopartial"    :  False,
    "price_loopleft"       :  False,
    "price_loopalternate"  :  False,
    "price_harristwopass"  :  False,
    "price_truenorminit"   :  False}

     
