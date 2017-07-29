#!/usr/bin/env python

import numpy as np
import os
import unittest

if __package__ is "" or "None": # py2 and py3 compatible 
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"
from .saveH5Object import SimpleH5Object, H5ObjectList, H5ObjectDict, RESERVED_VALUE_STRINGS, RESERVED_KEY_STRINGS

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


class DummyHolder(SimpleH5Object):

    def __init__(self, **kwargs):
        super(DummyHolder, self).__init__()

        for k, v in kwargs.items(): # inefficient in py2
            setattr(self, k, v)

class DummyHolderHolder(SimpleH5Object):

    def __init__(self, dummy_holders, **kwargs):
        sub_keys = ['dh_%s'%i for i in range(len(dummy_holders))]
        super(DummyHolderHolder, self).__init__(sub_keys=sub_keys)

        self.n_dummy_holders = len(dummy_holders)
        for i, dh in enumerate(dummy_holders):
            setattr(self, 'dh_%s'%(i), dh)

        for k, v in kwargs.items():  # inefficient in py2
            setattr(self, k, v)

    def h5_prepare_subs(self):
        for i in range(self.n_dummy_holders):
            setattr(self, 'dh_%s'%(i), DummyHolder())


class DataTester(BaseTest):

    def _test_items(self, items):
        c = DummyHolder(**items)
        c.save(TEST_FILE)
        c2 = DummyHolder(**{k: None for k in items.keys()})
        c2.load(TEST_FILE)
        for k in items.keys():
            v1 = getattr(c, k)
            v2 = getattr(c2, k)
            if type(v1) == dict:
                self.assertEqual(len(v1), len(v2))
                for k in v1.keys():
                    self.assertEqual(v1[k], v2[k])
            elif hasattr(v1, 'shape'):
                self.assertEqual(v1.shape, v2.shape)
            elif hasattr(v1, '__len__'):
                self.assertEqual(len(v1), len(v2))
                for i in range(len(v1)):
                    self.assertEqual(v1[i], v2[i])
            else:
                self.assertEqual(getattr(c, k), getattr(c2, k))

    def test_int(self):
        self._test_items({'zero': 0, 'one': 1, 'minus two': -2})

    def test_float(self):
        self._test_items({'zero': 0.0, 'one': 1.0, 'eps': 1.234e-15})

    def test_bool(self):
        self._test_items({'true': True, 'false': False})

    def test_string(self):
        self._test_items({'a': 'a', 'abc': 'abc', 'None': 'None'})

    def test_None(self):
        self._test_items({'None_type': None})

    def test_array(self):
        self._test_items({'number': np.array(1.2), 'empty1d': np.array([]),
                          '1d': np.arange(3), '2d': np.ones((3, 2, 4)),
                          'complex': np.exp(1.j*np.arange(3))})

    def test_list(self):
        self._test_items({'empty': [], 'assorted': ['string', 123, 0.52]})

    def test_tuple(self):
        self._test_items({'empty': (), 'assorted': ('string', 3, 0.5, None)})

    def test_dict(self):
        d = {'empty': {}, 'assorted': {'string': 'asdf', 'int': 1}}
        self._test_items(d)

class SubordinateTester(BaseTest):

    def test_subordinates(self):
        dh1 = DummyHolder(**{'a': 'abc', 'b': 0.3, 'c': -1})
        dh2 = DummyHolder(**{'d': 1234})
        dhh = DummyHolderHolder([dh1, dh2], **{'e': 321, 'f': 0.2})
        dhh.save(TEST_FILE)
        dhh2 = DummyHolderHolder([])
        dhh2.load(TEST_FILE)
        self.assertEqual(dhh.n_dummy_holders, dhh2.n_dummy_holders)
        dh1_2 = dhh2.dh_0
        dh2_2 = dhh2.dh_1
        self.assertEqual(dhh.e, dhh2.e)
        self.assertEqual(dhh.f, dhh2.f)
        for k in dh1.__dict__.keys():
            self.assertEqual(getattr(dh1, k), getattr(dh1_2, k))
        for k in dh2.__dict__.keys():
            self.assertEqual(getattr(dh2, k), getattr(dh2_2, k))
 
class ExceptionTester(BaseTest):

    def test_value_strings(self):
        for vstr in RESERVED_VALUE_STRINGS:
            with self.assertRaises(Exception):
                c = DummyHolder(asdf=vstr)
                c.save(TEST_FILE)
            _tear_down()

    def test_key_strings(self):
        for kstr in RESERVED_KEY_STRINGS:
            with self.assertRaises(Exception):
                kwargs = {kstr + 'asdf': 0.3}
                c = DummyHolder(**kwargs)
                c.save(TEST_FILE)

class H5ObjectListTester(BaseTest):

    def test_object_list(self):
        dhs = [DummyHolder(**{str(i): i*2}) for i in range(3)]
        h5ol = H5ObjectList(dhs)
        h5ol.save(TEST_FILE)
        h5ol2 = H5ObjectList([DummyHolder() for i in range(3)])
        h5ol2.load(TEST_FILE)
        for i in range(3):
            dh1 = dhs[i]
            dh2 = h5ol2[i]
            for k in dh1.__dict__.keys():
                self.assertEqual(getattr(dh1, k), getattr(dh2, k))

class H5ObjectDictTester(BaseTest):

    def test_object_list(self):
        dhs = {i: DummyHolder(**{str(i): i*2}) for i in range(1, 4)}
        h5ol = H5ObjectDict(dhs)
        h5ol.save(TEST_FILE)
        h5ol2 = H5ObjectDict({i: DummyHolder() for i in range(1, 4)})
        h5ol2.load(TEST_FILE)
        for i in range(1, 4):
            dh1 = dhs[i]
            dh2 = h5ol2[i]
            for k in dh1.__dict__.keys():
                self.assertEqual(getattr(dh1, k), getattr(dh2, k))

