#!/usr/bin/env python

import numpy as np
import os
import unittest

if __package__ is "" or "None": # py2 and py3 compatible 
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"
from .nodeFunction import DummyNodeFunction, Polyfit1D
from gwsurrogate import parametric_funcs as pf

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


class DummyNodeTester(BaseTest):

    def _verify_output(self, dnf, inputs, return_val):
        for x in inputs:
            y = dnf(x)
            if return_val is None:
                y2 = np.mean(x)
                self.assertEqual(y, y2)
            else:
                self.assertEqual(y, return_val)

    def _test(self, inputs, return_val=None):
        dnf = DummyNodeFunction(return_value=return_val)
        self._verify_output(dnf, inputs, return_val)
        dnf.save(TEST_FILE)

        dnf2 = DummyNodeFunction()
        dnf2.load(TEST_FILE)
        self._verify_output(dnf, inputs, return_val)
        _tear_down()

    def test_dummy_node_function(self):
        inputs = [[1.0], [2.0], np.array([3, 4.0]), np.ones((2, 3))]
        self._test(inputs, None)
        self._test(inputs, 1.2)


class Polyfit1DTester(BaseTest):

    def _verify_output(self, pf1d, inputs, outputs):
        for x, y in zip(inputs, outputs):
            yfit = pf1d([x])
            self.assertEqual(y, yfit)

    def _test(self, inputs, n_coefs, function_name):
        coefs = np.random.random(n_coefs)
        pf1d = Polyfit1D(function_name, coefs)
        func = pf.function_dict[function_name]
        outputs = [func(coefs, x) for x in inputs]
        self._verify_output(pf1d, inputs, outputs)

        pf1d.save(TEST_FILE)
        pf1d2 = Polyfit1D()
        pf1d2.load(TEST_FILE)
        self._verify_output(pf1d2, inputs, outputs)
        _tear_down()

    def test_all_functions(self):
        inputs = [0.1, 0.5]
        self._test(inputs, 1, 'polyval_1d')
        self._test(inputs, 5, 'polyval_1d')
        self._test(inputs, 3, 'ampfitfn1_1d')
        self._test(inputs, 3, 'ampfitfn2_1d')
        self._test(inputs, 4, 'phifitfn1_1d')
        #self._test(inputs, 5, 'ampfitfn3_1d')
        self._test(inputs, 4, 'ampfitfn4_1d')
        self._test(inputs, 5, 'nuSingularPlusPolynomial')
        self._test(inputs, 5, 'nuSingular2TermsPlusPolynomial')
