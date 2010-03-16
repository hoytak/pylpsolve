# PyLPSolve is an object oriented wrapper for the open source LP
# solver lp_solve. Copyright (C) 2010 Hoyt Koepke.
#
# This library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of
# the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA


# Sets the default options; this is also the authoritative list of
# what options are allowed and which aren't.

_default_options = {
    "presolve_none"        :   False,
    "presolve_rows"        :   False,
    "presolve_cols"        :   False,
    "presolve_lindep"      :   False,
    "presolve_aggregate"   :   False,
    "presolve_sparser"     :   False,
    "presolve_sos"         :   False,
    "presolve_reducemip"   :   False,
    "presolve_knapsack"    :   False,
    "presolve_elimeq2"     :   False,
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
    "price_truenorminit"   :  False,

    # Scaling stuff
    "scaling"              : "geometric",
    "scale_quadratic"      : False,
    "scale_logarithmic"    : False,
    "scale_userweight"     : False,
    "scale_power2"         : False,
    "scale_equilibrate"    : True,
    "scale_integers"       : False,
    "scale_dynupdate"      : False,
    "scale_rowsonly"       : False,
    "scale_colsonly"       : False,

    # Bookkeeping stuff
    "verbosity"            :  1,       # Lowest
    "error_on_bad_guess"   :  False
}

     
# Set the presolve flags
_presolve_flags = {
    "presolve_none"        : 0,
    "presolve_rows"        : 1,
    "presolve_cols"        : 2,
    "presolve_lindep"      : 4,
    "presolve_aggregate"   : 8,
    "presolve_sparser"     : 16,
    "presolve_sos"         : 32,
    "presolve_reducemip"   : 64,
    "presolve_knapsack"    : 128,
    "presolve_elimeq2"     : 256,
    "presolve_impliedfree" : 512,
    "presolve_reducegcd"   : 1024,
    "presolve_probefix"    : 2048,
    "presolve_probereduce" : 4096,
    "presolve_rowdominate" : 8192,
    "presolve_coldominate" : 16384,
    "presolve_mergerows"   : 32768,
    "presolve_impliedslk"  : 65536,
    "presolve_colfixdual"  : 131072,
    "presolve_bounds"      : 262144,
    "presolve_duals"       : 524288,
    "presolve_sensduals"   : 1048576}
   
_pricer_lookup = {
    "firstindex"           : 0,
    "danzig"               : 1,
    "devex"                : 2,
    "steepestedge"         : 3}

_pricer_flags = {
    "price_primalfallback" : 4,
    "price_multiple"       : 8,
    "price_partial"        : 16,
    "price_adaptive"       : 32,
    "price_randomize"      : 128,
    "price_autopartial"    : 512,
    "price_loopleft"       : 1024,
    "price_loopalternate"  : 2048,
    "price_harristwopass"  : 4096,
    "price_truenorminit"   : 16384}

# need to add in scaling options
_scaling_lookup = {
    "none"                 : 0,
    "extreme"              : 1,
    "range"                : 2,
    "mean"                 : 3,
    "geometric"            : 4,
    "curtisreid"           : 7}

_scaling_flags = {
    "scale_quadratic"      : 8,	 
    "scale_logarithmic"    : 16,
    "scale_userweight"     : 31,
    "scale_power2"         : 32,
    "scale_equilibrate"    : 64,
    "scale_integers"       : 128,
    "scale_dynupdate"      : 256,
    "scale_rowsonly"       : 512,
    "scale_colsonly"       : 1024}
    
