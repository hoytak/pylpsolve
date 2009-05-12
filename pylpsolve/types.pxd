from types import IntType, LongType, FloatType
from numpy import isscalar

ctypedef unsigned char ecode
ctypedef double real

############################################################
# Miscilaneous utility functions for resolving types

cdef inline bint isnumeric(v):
    t_v = type(v)
    
    if (t_v is IntType
        or t_v is LongType
        or t_v is FloatType):
        return True
    else:
        return isscalar(v)
