############################################################
# Miscilaneous utility functions for resolving types

from types import IntType, LongType, FloatType
from numpy import isscalar

cdef inline isnumeric(v):
    t_v = type(v)

    global IntType
    
    if (t_v is IntType
        or t_v is LongType
        or t_v is FloatType):
        return True
    else:
        return isscalar(v)

cdef inline isposint(v):
    t_v = type(v)

    if (t_v is IntType or t_v is LongType) and v >= 0:
        return True


cdef inline bint issize(v):
    t_v = type(v)

    if (t_v is IntType or t_v is LongType) and v >= 1:
        return True

cdef inline bint istuplelist(list l):

    for t in l:
        if type(t) is not tuple or len(<tuple>t) != 2:
            return False

    return True
        
cdef inline bint isnumericlist(list l):
    for t in l:
        if not isnumeric(t):
            return False

    return True

cdef inline bint is2dlist(list l):
    cdef int length = -1

    for ll in l:
        if not isnumericlist(ll):
            return False
        if length == -1:
            length = len(<list>ll)
        else:
            if length != len(<list>ll):
                return False
    else:
        return True
        
