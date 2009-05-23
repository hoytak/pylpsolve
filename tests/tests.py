#!/usr/bin/env python

import unittest, sys

if __name__ == '__main__':
    dtl = unittest.defaultTestLoader

    import test_blocks
    import test_minimal_lp
    import test_corner
    import test_tiny_lp
    import test_errorcatch
    import test_bounds
    import test_graphs
    import test_basis

    ts = unittest.TestSuite([
            dtl.loadTestsFromModule(test_blocks),
            dtl.loadTestsFromModule(test_bounds),
            dtl.loadTestsFromModule(test_minimal_lp),
            dtl.loadTestsFromModule(test_corner),
            dtl.loadTestsFromModule(test_tiny_lp),
            dtl.loadTestsFromModule(test_errorcatch),
            dtl.loadTestsFromModule(test_graphs),
            dtl.loadTestsFromModule(test_basis)
            ])

    if '--verbose' in sys.argv:
        unittest.TextTestRunner(verbosity=2).run(ts)
    else:
        unittest.TextTestRunner().run(ts)
