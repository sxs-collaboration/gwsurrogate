#!/usr/bin/env python

import numpy as np
import os
import unittest

if __package__ is "" or "None": # py2 and py3 compatible 
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"

from gwsurrogate.new import surrogate, nodeFunction

TEST_FILE = 'test.h5' # Gets created and deleted


def _tear_down():
    if os.path.isfile(TEST_FILE):
        os.remove(TEST_FILE)

def _set_up():
    # Don't overwrite this in case it's actually needed by something else
    if os.path.isfile(TEST_FILE):
        raise Exception("{} already exists! Please move or remove it.")


class BaseTest(unittest.TestCase):

    def setUp(self):
        _set_up()

    def tearDown(self):
        _tear_down()


class SplinterpTester(BaseTest):

    def setUp(self):
        super(SplinterpTester, self).setUp()
        xmin = 0.
        xmax = 10.
        n_sparse = 60
        n_dense = 237
        self.abs_tol = 1.e-4
        self.x_sparse = np.linspace(xmin, xmax, n_sparse)
        self.x_dense = np.linspace(xmin, xmax, n_dense)

    def test_real(self):
        y_sparse = np.sin(self.x_sparse)
        y_dense = np.sin(self.x_dense)
        y_interp = surrogate._splinterp(self.x_dense, self.x_sparse, y_sparse)
        self.assertLess(np.max(abs(y_interp - y_dense)), self.abs_tol)

    def test_complex(self):
        y_sparse = np.exp(1.j*self.x_sparse)
        y_dense = np.exp(1.j*self.x_dense)
        y_interp = surrogate._splinterp(self.x_dense, self.x_sparse, y_sparse)
        self.assertLess(np.max(abs(y_interp - y_dense)), self.abs_tol)

class ParamSpaceTester(BaseTest):

    def _test_ParamDim_nudge(self, tol, xmin, xmax):
        pd = surrogate.ParamDim("mass", xmin, xmax, rtol=tol/(xmax - xmin))
        self.assertEqual(pd.nudge(xmin - 0.9*tol), xmin + tol)
        self.assertEqual(pd.nudge(xmax + 0.9*tol), xmax - tol)
        with self.assertRaises(Exception):
            pd.nudge(xmin - 1.1*tol)
        with self.assertRaises(Exception):
            pd.nudge(xmax + 1.1*tol)
        for x in [xmin + tol, 0.5*(xmin + xmax), xmax - tol]:
            self.assertEqual(pd.nudge(x), x)

    def test_ParamDim(self):
        # Test nudge
        self._test_ParamDim_nudge(1.e-4, 0., 1.)
        self._test_ParamDim_nudge(1.e-4, 5., 123)

        # Test save/load
        pd = surrogate.ParamDim("Length in beardseconds", 0.1, 1.e5)
        pd.save(TEST_FILE)
        pd2 = surrogate.ParamDim("", 0, 1)
        pd2.load(TEST_FILE)
        self.assertEqual(pd2.name, "Length in beardseconds")
        self.assertEqual(pd2.min_val, 0.1)
        self.assertEqual(pd2.max_val, 1.e5)

    def _test_ParamSpace_nudge(self, xmins, xmaxs, tols):
        params = []
        for xmin, xmax, tol in zip(xmins, xmaxs, tols):
            pd =  surrogate.ParamDim("x", xmin, xmax, rtol=tol/(xmax - xmin))
            params.append(pd)
        ps = surrogate.ParamSpace("param space", params)

        # Nudge all
        xmin = np.array([x for x in xmins])
        xmax = np.array([x for x in xmaxs])
        for i in range(len(xmins)):
            xmin[i] -= 0.9 * tols[i]
            xmax[i] += 0.9 * tols[i]
        xmin = ps.nudge_params(xmin)
        xmax = ps.nudge_params(xmax)
        for i in range(len(xmins)):
            self.assertEqual(xmin[i], xmins[i] + tols[i])
            self.assertEqual(xmax[i], xmaxs[i] - tols[i])

        # Nudge one
        xmin = np.array([x for x in xmins])
        xmax = np.array([x for x in xmaxs])
        for i in range(len(xmins)):
            xmin[i] -= 0.9 * tols[i]
            xmax[i] += 0.9 * tols[i]
            xmin = ps.nudge_params(xmin)
            xmax = ps.nudge_params(xmax)
            self.assertEqual(xmin[i], xmins[i] + tols[i])
            self.assertEqual(xmax[i], xmaxs[i] - tols[i])

        # Check Exceptions
        for i in range(len(xmins)):
            xmin = np.array([x for x in xmins])
            xmax = np.array([x for x in xmaxs])
            xmin[i] -= 1.1*tols[i]
            xmax[i] += 1.1*tols[i]
            with self.assertRaises(Exception):
                ps.nudge_params(xmin)
            with self.assertRaises(Exception):
                ps.nudge_params(xmax)

        # Check unmodified
        x = np.array([x + tol for x, tol in zip(xmins, tols)])
        check_params = [1*x]
        for i in range(len(xmins)):
            x[i] += 0.5*(xmaxs[i] - xmins[i]) - tols[i]
            check_params.append(1*x)
        for i in range(len(xmins)):
            x[i] += 0.5*(xmaxs[i] - xmins[i]) - tols[i]
            check_params.append(1*x)
        for x in check_params:
            x2 = ps.nudge_params(x)
            for xi, xi2 in zip(x, x2):
                self.assertEqual(xi, xi2)
        return ps

    def test_ParamSpace(self):
        ps = self._test_ParamSpace_nudge([0.], [1.], [1.e-4])
        ps = self._test_ParamSpace_nudge([1., 2., 100.],
                                         [1.1, 100., 101.],
                                         [1.e-4, 1.e-5, 1.e-6])

        # Test saving/loading
        ps.save(TEST_FILE)
        ps2 = surrogate.ParamSpace("", [])
        ps2.load(TEST_FILE)
        self.assertEqual(ps.name, ps2.name)
        self.assertEqual(ps.dim, ps2.dim)
        for i in range(ps.dim):
            pd = ps._params[i]
            pd2 = ps2._params[i]
            self.assertEqual(pd.min_val, pd2.min_val)
            self.assertEqual(pd.max_val, pd2.max_val)
            self.assertEqual(pd.tol, pd2.tol)

class SingleFunctionSurrogateTester(BaseTest):

    def test_SingleFunctionSurrogate_NoChecks(self):
        ei = np.array([np.ones(10), np.linspace(3., 6., 10)])
        node_functions = []
        for i in range(2):
            dummy = nodeFunction.DummyNodeFunction()
            node_functions.append(nodeFunction.NodeFunction('node_%s'%(i),
                    node_function=dummy))
        sfs_nc = surrogate._SingleFunctionSurrogate_NoChecks('test', ei,
                                                             node_functions)
        res = sfs_nc([1.0])
        for r, t in zip(res, np.linspace(3., 6., 10)):
            self.assertEqual(r, 1. + t)

        res = sfs_nc(1.0)
        for r, t in zip(res, np.linspace(3., 6., 10)):
            self.assertEqual(r, 1. + t)

        sfs_nc.save(TEST_FILE)
        sfs_nc2 = surrogate._SingleFunctionSurrogate_NoChecks('', np.array([]),
                                                              [])
        sfs_nc2.load(TEST_FILE)
        self.assertEqual(np.max(abs(sfs_nc2.ei_basis - sfs_nc.ei_basis)), 0.)

    def test_SingleFunctionSurrogate(self):
        pd = surrogate.ParamDim('mass', 1., 2.)
        ps = surrogate.ParamSpace('params', [pd])
        t = np.linspace(0., 10., 10)
        ei = np.array([np.ones(10), 1*t])
        nf = []
        for i in range(2):
            dummy = nodeFunction.DummyNodeFunction()
            nf.append(nodeFunction.NodeFunction('node_%s'%(i),
                    node_function=dummy))
        sfs = surrogate.SingleFunctionSurrogate('s', t, ps, ei, nf)

        def check(x, x_nudged, sur):
            res = sur(x)
            ans = x_nudged*(1 + t)
            for r, a in zip(res, ans):
                self.assertEqual(r, a)

        def check_cases(sur):
            check(1.0, 1.0 + 1.e-12, sur)
            check(np.array([1.0]), 1.0 + 1.e-12, sur)
            check(1.0 - 1.e-13, 1.0 + 1.e-12, sur)

        check_cases(sfs)

        sfs.save(TEST_FILE)
        sfs2 = surrogate.SingleFunctionSurrogate()
        sfs2.load(TEST_FILE)

        check_cases(sfs2)
