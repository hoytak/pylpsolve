"""
An accompaning module that contains verious functions for operations
on graphs.  
"""

from pylpsolve import LPSolve
from numpy cimport ndarray as ar
from numpy import int32,uint32,int64, uint64, float32, float64,\
    uint, empty, ones, zeros, uint, arange, isscalar, amax, amin, \
    ndarray, array, asarray, argsort, nonzero

from typeconfig import npint, npuint, npfloat

# This is how it should work; don't know why it doesn't
#from typechecks cimport *

# this works

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
        


########################################
# Set the proper types


cpdef graphCut(graph, int source, int sink):
    """
    Performs a graph cut on the graph specified by `graph`, returning
    a labeling of the vertices.  The vertices are integers, with ``0 <= vertex < nV``.

    `graph` may be specified in several ways:

    - A three tuple of 1d arrays or lists, of the form
      ``(origin_indices, dest_indices, capacity)``.

    - A dictionary with keys being 2-tuples (source_vertex,
      dest_vertex), and the values being the capacity.

    - A 2d double array / list of size nV x nV, where element (i,j)
      indicates the capacity of the edge from node i to node j.  If an
      element is 0, it indicates the edge is not connected.

    The graph is assumed to consist only of the edges connected in
    this graph.

    Note: the capacity on any edge must be >= 0.

    The return labeling is an integer array of length nV, with 1
    indicating a node is attached to the source, 0 indicating a node
    is attached to the sink, and -1 indicating it isn't connected to
    either the source or sink.
    """

    cdef ar[unsigned int] S, D
    cdef ar[double] C
    cdef size_t i
    cdef tuple t
    cdef int si, di

    if type(graph) is ndarray:
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Graph specified by array must be 2d and square.")

        if (graph < 0).any():
            raise ValueError("Negative edge capacities not allowed.")

        nz = nonzero(graph)

        return graphCutSparse(nz[0], nz[1], graph[nz], source, sink)

    elif type(graph) is list:
        if not is2dlist(graph) or len(graph) != len(graph[0]):
            raise ValueError("Graph specified by list of lists must be 2d and square.")
 
        # bump it back
        return graphCut(array(graph, npfloat), source, sink)
        
    elif type(graph) is tuple:
        t = <tuple>graph

        if len(<tuple>graph) != 3:
            raise ValueError("Graph specified by tuple must have three elements: (origin_vertices, dest_vertices, capacitys)")

        S = asarray(t[0], npuint)
        D = asarray(t[1], npuint)
        C = asarray(t[2], npfloat)

        return graphCutSparse(S, D, C, source, sink)
    elif type(graph) is dict:
        n = len(graph)
        S = empty(n, npuint)
        D = empty(n, npuint)
        C = empty(n, npfloat)

        for i, ((si,di), c) in enumerate((<dict>graph).iteritems()):
            if c < 0:
                raise ValueError("Negative edge capacities not allowed (%f)." % c)
            
            if si < 0: 
                raise ValueError("Negative vertex indices not allowed (%d)" % si)

            if di < 0: 
                raise ValueError("Negative vertex indices not allowed (%d)" % di)

            S[i] = si
            D[i] = di
            C[i] = c
            
        return graphCutSparse(S, D, C, source, sink)

cdef graphCutSparse(ar origin_indices, ar dest_indices, ar capacity, 
                    int source, int sink):
    """
    Performs a graph cut on the graph defined by `origin_indices`,
    `dest_indices`, and `capacity`. 

    """

    # Sort the imput sources, cause that makes several parts of the
    # algorithms easier
    idxmap = argsort(origin_indices)

    if not origin_indices.ndim == 1:
        raise ValueError("origin_indices must be 1d array.")

    if not dest_indices.ndim == 1:
        raise ValueError("dest_indices must be 1d array.")

    if not capacity.ndim == 1:
        raise ValueError("capacity must be 1d array.")

    if (capacity < 0).any():
        raise ValueError("Capacity must be >= 0")

    if not (origin_indices.shape[0] == dest_indices.shape[0] == capacity.shape[0]):
        raise ValueError("origin_indices, dest_indices, and capacity arrays must be the same length.")

    cdef ar[unsigned int, mode="c"] S    = asarray(origin_indices[idxmap], npuint)
    cdef ar[unsigned int, mode="c"] D    = asarray(dest_indices[idxmap], npuint)
    cdef ar[double, mode="c"] C          = asarray(capacity[idxmap], npfloat)

    # Now eliminate duplicates and 0-capacity edges
    elim_idx = (S == D) | (C == 0)
    if elim_idx.any():
        keep_idx = ~elim_idx
        S = S[keep_idx]
        D = D[keep_idx]
        C = C[keep_idx]

    cdef size_t nE = S.shape[0], nV = max(S[-1], D.max()) + 1

    if source > nV or source < 0:
        raise ValueError("source index not valid.")

    if sink > nV or sink < 0:
        raise ValueError("Sink index not valid.")
    
    lp = LPSolve()

    vblock = lp.getVariables("p", nV)

    assert vblock == (0, nV)

    xblock = lp.getVariables("x", nE)

    assert xblock == (nV, nV + nE)

    cdef ar[double, mode="c"] w 
    cdef ar[int, mode="c"]    idx


    ############################################################
    # First set the p on the dual
    # p_v - p_u + x_uv \geq 0
    
    w = empty(3)
    w[0], w[1], w[2] = 1, -1, 1
    
    idx = empty(3, npint)

    for 0 <= i < nE:
        idx[0] = D[i]
        idx[1] = S[i]
        idx[2] = nV + i

        lp.addConstraint( (idx, w), ">=", 0)

    
    ########################################
    # Next set the source and node constraint

    lp.addConstraint( {source : 1, sink : -1}, ">=", 1)

    ########################################
    # Now bound the various variables

    lp.setUnbounded(vblock)
    lp.setLowerBound(xblock, 0)
    
    ########################################
    # Set the objective function

    lp.setObjective( (xblock, C) )

    lp.solve()

    cdef ar[double, mode="c"] x = lp.getSolution(xblock)
    
    lp.clear()

    ############################################################
    # Convert solution into labelings of the nodes.  

    # Set up the labels
    cdef ar[int, mode="c"] labels = empty(nV, npint)
    labels[:] = -1  # not set

    ##############################
    # First label all the ones in the source, up to the graph cut

    # construct a mapping on the S indices
    cdef ar[unsigned int, mode="c"] mapping = make_mapping(S, nV, nE)

    labelGraphSource(<int*>labels.data, <unsigned int*> mapping.data,
                     <int*>S.data, <int*>D.data, <double*> x.data, source, nE)

    ########################################
    # Now go backwards from sink to source, up to the graph cut

    # Go back to the nodes
    D_idxmap = argsort(D)
    S = S[D_idxmap]
    D = D[D_idxmap]
    x = x[D_idxmap]
    
    mapping = make_mapping(D, nV, nE)

    labelGraphSink(<int*>labels.data, <unsigned int*> mapping.data,
                    <int*>S.data, <int*>D.data, <double*>x.data, sink, nE)
    
    return labels


cdef inline ar make_mapping(ar A_o, size_t nV, size_t nE):
  
    cdef ar[unsigned int, mode="c"] A = A_o
    cdef size_t i
    
    cdef ar[unsigned int, mode="c"] mapping = empty(nV, npuint)

    for 0 <= i < nV:  
        mapping[i] = nE  # I.e. not set; not used above

    for 0 <= i < nE:
        if mapping[A[i]] == nE:
            mapping[A[i]] = i

    return mapping



# Easiest to do this recursively
    
cdef labelGraphSource(int *labels, unsigned int *mapping_S, 
                      int *S, int *D, double *x,
                      int set_node, size_t nE):

    labels[set_node] = 1

    cdef unsigned int cur_edge = mapping_S[set_node]

    if cur_edge == nE:  # Not connected to anything
        return

    cdef int dest_label

    while S[cur_edge] == set_node:

        dest_label = labels[D[cur_edge]]

        if x[cur_edge] == 0:
            # It's not part of the cut
            
            assert dest_label in [-1, 1]

            if dest_label == -1:
                labelGraphSource(labels, mapping_S, S, D, x, D[cur_edge], nE)

        cur_edge += 1


cdef labelGraphSink(int *labels, unsigned int *mapping_D, 
                    int *S, int *D, double *x,
                    int set_node, size_t nE):

    labels[set_node] = 0

    # work backwards, now on D -> S
    cdef unsigned int cur_edge = mapping_D[set_node]

    if cur_edge == nE:  # Not connected to anything
        return

    cdef int dest_label

    while D[cur_edge] == set_node:
        
        dest_label = labels[S[cur_edge]]

        if x[cur_edge] == 0:
            # It's not part of the cut
            
            assert dest_label in [-1, 0]

            if dest_label == -1:
                labelGraphSink(labels, mapping_D, S, D, x, S[cur_edge], nE)

        cur_edge += 1
