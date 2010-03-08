"""
An accompaning module that contains verious functions for operations
on graphs.  
"""

from pylpsolve import LP
from numpy cimport ndarray as ar
from numpy import int32,uint32,int64, uint64, float32, float64,\
    uint, empty, ones, zeros, uint, arange, isscalar, amax, amin, \
    ndarray, array, asarray, argsort, nonzero

from typeconfig import npint, npuint, npfloat

# This is how it should work; don't know why it doesn't
#from typechecks cimport *

# this works

cdef double eps = 10e-12

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

cdef inline bint isnumerictuple(tuple t):
    for e in t:
        if not isnumeric(e):
            return False

    return True

cdef inline bint isposintlist(list l):
    for t in l:
        if not isposint(t):
            return False

    return True

cdef inline bint is2dlist(list l, bint tuple_okay):
    cdef int length = -1

    for ll in l:
        if tuple_okay:
            if not ((type(ll) is list and isnumericlist(ll)) 
                    or (type(ll) is tuple and isnumerictuple(ll))):
                return False
        else:
            if not (type(ll) is list and isnumericlist(ll)):
                return False

        if length == -1:
            length = len(<list>ll)
        else:
            if length != len(<list>ll):
                return False

    else:
        return True
        

################################################################################
# Graph Cuts

def graphCut(graph, int source, int sink):
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
        if not is2dlist(graph, False) or len(graph) != len(graph[0]):
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

    # Normalize the C so the cut we do for numerical stability works
    C /= C.mean()

    cdef size_t nE = S.shape[0], nV = max(S[-1], D.max()) + 1

    if source > nV or source < 0:
        raise ValueError("source index not valid.")

    if sink > nV or sink < 0:
        raise ValueError("Sink index not valid.")
    
    lp = LP()

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

    lp.solve(scale_integers=False,scaling="extreme")

    cdef ar[double, mode="c"] x = lp.getSolution(xblock)
    
    lp.clear()

    ############################################################
    # Convert solution into labelings of the nodes.  

    # Set up the labels
    cdef ar[int, mode="c"] labels = empty(nV, npint)
    labels[:] = -1  # not set

    ##############################
    # First label all the ones in the source, up to the graph cut Then
    # go backwards from sink to source, up to the graph cut. 

    # Working S -> D
    # construct a mapping on the S indices

    cdef ar[unsigned int, mode="c"] mapping = make_mapping(S, nV, nE)

    labelGraph(<int*>labels.data, <unsigned int*> mapping.data,
                <int*>S.data, <int*>D.data, <double*> x.data, source, nE, nV, 1)

    # Working D -> S
    D_idxmap = argsort(D)
    S = S[D_idxmap]
    D = D[D_idxmap]
    C = C[D_idxmap]
    x = x[D_idxmap]
    
    mapping = make_mapping(D, nV, nE)

    labelGraph(<int*>labels.data, <unsigned int*> mapping.data,
                <int*>D.data, <int*>S.data, <double*>x.data, sink, nE, nV, 0)
    
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
cdef labelGraph(int *labels, unsigned int *mapping_S, 
                int *S, int *D, double *x,
                int set_node, size_t nE, size_t nV, int label):

    assert 0 <= set_node < nV

    if labels[set_node] == label:
        return

    labels[set_node] = label

    cdef unsigned int cur_edge = mapping_S[set_node]

    cdef int dest_label

    while cur_edge != nE and S[cur_edge] == set_node:

        assert 0 <= D[cur_edge] < nV, "D[cur_edge] = %d; nV = %d" % (D[cur_edge], nV)

        dest_label = labels[D[cur_edge]]

        if x[cur_edge] <= eps:
            # It's not part of the cut
            assert dest_label in [-1, label]
            
            labelGraph(labels, mapping_S, S, D, x, D[cur_edge], nE, nV, label)

        cur_edge += 1


################################################################################
# Potential Function Potentials

def maximizeGraphPotential(E1, E2):
    """
    Maximizes a potential function defined by weights on nodes and
    interactions between nodes.  The nodes may be either 0 or 1, and
    the potential function must be of the form:

    .. math::
       \sum_{i,j} E_{ij}(v_i, v_j) + \sum_{i} E_i(v_i)
    
    where E_i(v_i) is given by `E1` and E_{ij} is specified by `E2`.

    Additionally, E_{ij}(v_i, v_j) must satisfy the following
    regularity condition:

    .. math::
       E_{ij}(0,1) + E_{ij}(1,0) \leq E_{ij}(0,0) + E_{ij}(1,1)

    The problem is NP-hard in the number of energy potentials that do
    not satisfy this constructions (See [REF]).

    `E1` must be a 1d vector or list giving the weights on the nodes
    being on.  Note that the problem is not changed by scaling with a
    constant factor, so the difference between energies of a node
    being on and off is sufficient for the problem.

    `E2` can be:

    - A tuple of 3 arrays or lists.  

      The first arrays/lists must be 1d with respective elements
      giving the two vertices of the edge.  If they are given as a
      single array, then it must be n x 2, with the vertices given in
      the second element.

      The third array/list specifies the 4 values of the potential
      function in the order (0,0), (0,1), (1,0), (1,1).  Thus it must
      either be a list of lists or tuples, or a 2d array, with the
      second/inner dimension specifying the 4 elements.

    - A dictionary. with tuples for keys and lists, tuples, or arrays
      for the values.  In this case, the keys must be 2-tuples
      specifying the two vertices.  The values must be lists, tuples,
      or arrays of length 4 giving the 4 values of the potential
      function.

    - A nV x nV x 4 array, where the (i,j) 4-vector specifies the
      potential function between the ith and jth vertices.  Only the
      upper triangle, i < j, is considered, as the matrix should be
      symmetric.

    The value returned is 1d array with 0 or 1 giving the best value
    of the potential function, or -1 if it doesn't matter.
    """

    cdef size_t i, ii
    cdef tuple t
    cdef list l
    cdef ar a1, a2, a3
    cdef ar[unsigned int, mode="c"] va1, va2
    cdef ar[int, mode="c"] va1i, va2i
    cdef ar[double, ndim=2] E2a
    cdef ar[double, mode="c"] E1a
    cdef dict d
    cdef ar[double, ndim=3] A3

    cdef bint comp_0, comp_eq
    cdef unsigned int v1, v2

    ########################################
    # Validate E1

    if type(E1) is list:
        if not isnumericlist(E1):
            raise ValueError("E1 must be 1d array or numeric list.")

        E1a = array(E1, npfloat)

    elif type(E1) is ndarray:
        a1 = (<ar>E1)

        if a1.ndim != 1:
            raise ValueError("E1 must be 1d array or numeric list.")

        E1a = asarray(E1, npfloat).ravel()
    elif E1 is None:
        E1a = None
    else:
        raise ValueError("E1 must be 1d array or numeric list.")
        

    ########################################
    # Validate E2

    if type(E2) is tuple:
        t = <tuple>E2

        if len(t) != 3:
            raise ValueError("E2 must be specified by a 3-tuple or dict.")
        
        if type(t[0]) is list:
            l = <list>(t[0])
            if not isposintlist(l):
                raise ValueError("First vertex list must be list of positive integers.")
                
            va1 = array(l, npuint)

        elif type(t[0]) is ndarray:
            a1 = t[0]
            
            if a1.ndim != 1:
                raise ValueError("First vertex array must be 1d")

            va1 = asarray(a1, npuint).ravel()
        else:
            raise ValueError("Second vertex list must be either 1d array or list.")

        if type(t[1]) is list:
            l = <list>(t[1])
            if not isposintlist(l):
                raise ValueError("Second vertex list must be list of positive integers.")
                
            va2 = array(l, npuint)
        elif type(t[1]) is ndarray:
            a2 = t[1]
            
            if a2.ndim != 1:
                raise ValueError("Second vertex array must be 1d")

            va2 = asarray(a2, npuint).ravel()
        else:
            raise ValueError("Second vertex list must be either 1d array or list.")
        
        # Now validate the following 
        if type(t[2]) is ndarray:
            a3 = asarray(t[2], npfloat)

            if a3.ndim != 2:
                raise ValueError("Potentail function potential array must be 2d")
            
            if a3.shape[1] != 4:
                raise ValueError("Second dimension of potential function array must have size 4.")
            
            E2a = a3
        elif type(t[2]) is list:
            if not is2dlist(<list>t[2], True):
                raise ValueError("Potential function must be specified by lists of lists or tuples, or a 2d array.")
            
            E2a = array(t[2], npfloat)
        else:
            raise ValueError("Potential function must be specified by lists of lists or tuples, or a 2d array.")

    elif type(E2) is dict:
        d = <dict>E2
        
        n = len(d)
        
        va1 = empty(n, npuint)
        va2 = empty(n, npuint)
        E2a = empty( (n, 4), npfloat)

        for i, (k, v) in enumerate(d.iteritems()):
            if type(k) is not tuple or len(<tuple>k) != 2:
                raise ValueError("Keys in potential function dictionary must be 2-tuples specifying vertices.")
            
            if not (type(v) is tuple or type(v) is list) or len(v) != 4:
                raise ValueError("Values in potential function dictionary must be lists or tuples of size 4.")
            
            v1, v2 = <tuple>k

            if not (isposint(v1) and isposint(v2)):
                raise ValueError("Indices in key must be positive integers.")
            
            va1[i] = v1
            va2[i] = v2
            
            if not (isnumeric(v[0]) and isnumeric(v[1]) and isnumeric(v[2]) and isnumeric(v[3])):
                raise ValueError("Potential function values must be numeric.")

            E2a[i, 0] = v[0]
            E2a[i, 1] = v[1]
            E2a[i, 2] = v[2]
            E2a[i, 3] = v[3]

    elif type(E2) is ndarray:
        a1 = <ar>E2

        if a1.ndim != 3 or a1.shape[2] != 4 or a1.shape[0] != a1.shape[1]:
            raise ValueError("Potential function array must be nV x nV x 4.")

        A3 = a1

        # Convert it to the proper sparse interpretation
        va1i, va2i = nonzero( (A3[:,:,0] != 0) | (A3[:,:,1] != 0) 
                              | (A3[:,:,2] != 0) | (A3[:,:,3] != 0) )
        
        # Now we need to check that if (i,j) is valid, then element
        # (j,i) is either 0 or equal:

        va1 = empty(va1i.shape[0], npuint)
        va2 = empty(va1i.shape[0], npuint)
        ii = 0

        for 0 <= i < va1i.shape[0]:
            v1 = va1i[i]
            v2 = va2i[i]

            if v1 == v2: 
                continue

            comp_0  = (A3[v2,v1,0] == A3[v2,v1,1] == A3[v2,v1,2] == A3[v2,v1,3] == 0)
            comp_eq = (A3[v2,v1,0] == A3[v1,v2,0] and A3[v2,v1,1] == A3[v1,v2,1]
                        and A3[v2,v1,2] == A3[v1,v2,2] and A3[v2,v1,3] == A3[v1,v2,3])

            if not (comp_0 or comp_eq):
                raise ValueError("Matrix form of E2 must be either symmetric or only one of E2[i,j,:] and E2[j,i,:] must be nonzero (i=%d,j=%d)." % (v1, v2))
            
            if comp_0 or comp_eq and v1 < v2:
                va1[ii] = v1
                va2[ii] = v2
                ii += 1

        # Trim
        va1 = va1[:ii]
        va2 = va2[:ii]

        E2a = empty( (va1.shape[0], 4), npfloat)

        for 0 <= i < va1.shape[0]:
            E2a[i, 0] = A3[va1[i], va2[i], 0]
            E2a[i, 1] = A3[va1[i], va2[i], 1]
            E2a[i, 2] = A3[va1[i], va2[i], 2]
            E2a[i, 3] = A3[va1[i], va2[i], 3]

    elif E2 is None:
        if E1 is None:
            raise ValueError("E1 and E2 cannot both be None.")
        
        ret = empty(E1a.shape[0], npint)
        
        ret[E1a > 0] = 1
        ret[E1a < 0] = 0
        ret[E1a == 0] = -1

        return ret

    else:
        raise ValueError("E2 must be either 3-tuple or dict.")


    # We now have E1a, va1, va2, and E2a.  Now create the Ising model.

    # first create an array of 
    
    if not (va1.shape[0] == va2.shape[0] == E2a.shape[0]):
        raise ValueError("Index arrays and potential function specifications must be same length.")

    cdef size_t nV = max(va1.max(), va2.max(), 0 if E1a is None else E1a.shape[0] - 1) + 1
    cdef size_t nE = va1.shape[0]

    if E1a is not None and E1a.shape[0] != nV:
        raise ValueError("Length of E1 (%d) must equal the number of vertices (%d), as deduced from labels in E2." % (E1a.shape[0], nV))

    # Create the edge lists; see REF for details

    cdef ar[unsigned int, mode="c"] S = empty(nE + 2*nV, npuint)
    cdef ar[unsigned int, mode="c"] D = empty(nE + 2*nV, npuint)
    cdef ar[double, mode="c"]       C = zeros(nE + 2*nV, npfloat)

    cdef size_t source = nV
    cdef size_t sink   = nV + 1

    # Set up the indices regarding source and sink
    for 0 <= i < nV:
        S[nE + i] = source
        D[nE + i] = i

        S[nE + nV + i] = i
        D[nE + nV + i] = sink 

    # Set up the individual energy functions
    if E1a is not None:
        for 0 <= i < nV:
            if E1a[i] < 0:
                # node from vertex to sink
                C[nE + nV + i] += -E1a[i] 
            elif E1a[i] > 0:
                # Node from source to vertex
                C[nE + i] += E1a[i]
    
    cdef double w, CmA, CmD

    # Set up the inter-node energy functions
    for 0 <= i < nE:

        CmA = E2a[i, 0] - E2a[i, 2]   # Reversed from the paper
        CmD = E2a[i, 3] - E2a[i, 2]

        w = -E2a[i, 1] - E2a[i,2] + E2a[i,0] + E2a[i,3]

        if w < 0:
            raise ValueError("Potential function of edge %d is not regular." % i)

        C[i] += w
        S[i] = va1[i] 
        D[i] = va2[i]

        # Now we need to set the graph so it flows from source to sink
        if CmA > 0:
            C[nE + S[i]] += CmA
        elif CmA < 0:
            C[nE + D[i]] += -CmA

        if CmD > 0:
            C[nE + nV + D[i]] += CmD
        else:
            C[nE + nV + S[i]] += -CmD

    return graphCutSparse(S,D,C,source,sink)[:-2]



