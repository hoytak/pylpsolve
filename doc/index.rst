PyLPSolve
========================================

PyLPSolve is an object oriented wrapper for the open source LP solver
`lpsolve`_.  The focus is on usability and integration with existing
python packages used for scientific programming (i.e. numpy and
scipy).

One unique feature is a convenient bookkeeping system that allows the
user to specifiy blocks of variables by string tags, or other index
block methods, then work with these blocks instead of individual
indices.  All the elements of the LP are cached until solve is called,
with memory management and proper sizing of the LP in lpsolve handled
automatically.

PyLPSolve is written in cython, with all low-level processing done
effectively in low-level C for speed.  Thus there should be mimimal
overhead to using this wrapper.  It is licensed under the liberal BSD
open source license (though LPSolve is licensed under the LGPLv2
license).

PyLPSolve Distinctives
------------------------------

- A design emphasis on usability and reliability.

- Many bookkeeping operations are automatically handled by abstracting
  similar variables into blocks that can be handled as a unit with
  arrays or matrices.

- LP sizing is handled automatically; a buffering system ensures this
  is fast and usable.

- Full integration with numpy arrays.

- Written in `Cython`_ for speed; all low-level operations are done in
  compiled and optimized C code.

- Good coverage by test cases.

- Licensed under the LGPL open source license (as is LPSolve_).

Short Example
------------------------------

Consider the following simple linear program:

  .. math::
    \begin{array}{lr} 
    \operatorname{maximize } & x + y + z \\
    \text{subject to} & x + y \leq 3 \\ 
    \quad & y + 2z \leq 4 
    \end{array}
    
This can be specified by the following code::

    lp = LP()
    
    # Specify constraints
    lp.addConstraint([[1,1,0], [0,1,2]], "<=", [3, 4])

    # set objective    
    lp.setObjective([1,1,1], mode="maximize")
    
    # Run
    lp.solve()

    # print out the solution:
    print lp.getSolution()

which would print the solution::
   
   [3, 0, 2]

This is the simplest way to work with constraints; numerous other ways
are possible.  For example, this code is equivalent to the above::

    lp = LP()
    
    # Specify constraints
    lp.addConstraint({"x" : 1, "y" : 1}, "<=", 3)
    lp.addConstraint({"y" : 1, "z" : 2}, "<=", 4)

    # set objective    
    lp.setObjective({"x": 1, "y" : 1, "z" : 1}, mode="maximize")
    
    # Run
    lp.solve()

    # print out the solution:
    print lp.getSolution()

Numerous other ways of working with constraints and named blocks of
variables are possible.  For more examples and information, see the
:ref:`api`.


Contents
========================================

.. toctree::
    :maxdepth: 3
    
    api
    download
    license

Authors
========================================

The PyLPSolve wrapper was written by `Hoyt Koepke`_, building on the
work of the lpsolve_ team.  Contributions are welcome.

.. include:: references.rst 
