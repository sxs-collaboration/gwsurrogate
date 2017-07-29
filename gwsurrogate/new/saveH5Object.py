"""Base class for loading and saving in h5 format"""

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import h5py
import numpy as np
import os

NONE_STR = "NONE_TYPE"
RESERVED_VALUE_STRINGS = [NONE_STR]

DICT_PREFIX = "DICT_"
LIST_PREFIX = "LIST_"
TUPLE_PREFIX = "TUPLE_"
OBJ_DICT_KEY_STR = "KEYS" # Does not need to be reserved
RESERVED_KEY_STRINGS = [DICT_PREFIX, LIST_PREFIX, TUPLE_PREFIX]


def _ensure_not_reserved(k, v):
    if type(v) == str and v in RESERVED_VALUE_STRINGS:
        raise Exception("The string %s is reserved"%(v))
    if type(k) == str:
        for rk in RESERVED_KEY_STRINGS:
            if k[:len(rk)] == rk:
                raise Exception("Keys may not begin with %s, got %s"%(rk, k))


def _list_item_string(i):
    return 'ITEM_' + str(i)


def _write_attr(f, k, v):
    """
    Writes a single attribute to h5 format, allowing for None, nested lists
    and/or dictionaries.
    Arguments:
        f: An h5py file or group to which this attribute will be written.
           Lists and dictionaries are written as groups, everything else
           is written as a dataset.
        k: The attribute name (a string)
        v: The attribute value. Should be None, a list, a dictionary, or any
           format supported by h5py (for example numpy arrays). Nested elements
           must also be of one of these formats.
    """
    _ensure_not_reserved(k, v)
    if v is None:
        f.create_dataset(k, data=NONE_STR)
    elif type(v) == dict:
        g = f.create_group(DICT_PREFIX + k)
        for kk, vv in v.items():  # inefficient in py2
            _write_attr(g, kk, vv)
    elif type(v) == list:
        g = f.create_group(LIST_PREFIX + k)
        for i in range(len(v)):
            _write_attr(g, _list_item_string(i), v[i])
    elif type(v) == tuple:
        g = f.create_group(TUPLE_PREFIX + k)
        for i in range(len(v)):
            _write_attr(g, _list_item_string(i), v[i])
    else:
        f.create_dataset(k, data=v)


def _read_attrs(f):
    """
    Reads all attributes which were written using _write_attr.
    Arguments:
        f: An h5py file or group
    """
    d = {}
    for k, item in f.items():  # inefficient in py2
        if k[:len(DICT_PREFIX)] == DICT_PREFIX:
            d[k[len(DICT_PREFIX):]] = _read_attrs(item)
        elif k[:len(LIST_PREFIX)] == LIST_PREFIX:
            tmp_d = _read_attrs(item)
            v = [tmp_d[_list_item_string(i)] for i in range(len(tmp_d))]
            d[k[len(LIST_PREFIX):]] = v
        elif k[:len(TUPLE_PREFIX)] == TUPLE_PREFIX:
            tmp_d = _read_attrs(item)
            v = tuple(tmp_d[_list_item_string(i)] for i in range(len(tmp_d)))
            d[k[len(TUPLE_PREFIX):]] = v
        elif type(item) == h5py._hl.dataset.Dataset:
            v = item.value
            if type(v) == np.string_:
                v = str(v)
            if type(v) == str and v == NONE_STR:
                d[k] = None
            else:
                d[k] = v
    return d


class SimpleH5Object(object):
    """A simple base class that can save and load itself using h5 files"""

    def __init__(self, data_keys=None, sub_keys=[]):
        """
        If data_keys is None, the other arguments have no effect and each
        element of self.__dict__ is stored as an h5 dataset.

        Otherwise:
            data_keys: attribute names which should be saved as datasets
            sub_keys: attribute names which are SimpleH5Object instances.
                      Class instances must be generated in h5_prepare_subs
                      so they can load their own data.
        """
        self._h5_data_keys = data_keys
        self._h5_subordinate_keys = sub_keys

    def save(self, filename):
        """Save data to h5 file"""
        if os.path.exists(filename):
            raise Exception("Will not overwrite %s"%(filename))
        with h5py.File(filename, 'w') as f:
            self._write_h5(f)

    def load(self, filename):
        """Load data from h5 file"""
        with h5py.File(filename, 'r') as f:
            self._read_h5(f)

    def _default_data_keys(self):

        # removed by 2to3 tool
        #keys = filter(lambda s: s not in self._h5_subordinate_keys,
        #              self.__dict__.keys())
        keys = [s for s in list(self.__dict__.keys()) if s not in self._h5_subordinate_keys]
        return keys

    def _write_h5(self, f):
        """Write data to h5 File or group f."""
        if self._h5_data_keys is None:
            keys = self._default_data_keys()
        else:
            keys = self._h5_data_keys
        self._write_data(f, keys)
        self._write_subordinates(f)

    def _write_data(self, f, keys):
        for k in keys:
            #print k
            # TODO: how to block auto-save mechanism?
            if k != "last_return":
              v = getattr(self, k)
              _write_attr(f, k, v)

    def _write_subordinates(self, f):
        for k in self._h5_subordinate_keys:
            #print k
            g = f.create_group(k)
            getattr(self, k)._write_h5(g)

    def h5_prepare_subs(self):
        """Override this to do setup subordinate objects before loading them"""
        return

    def _read_h5(self, f):
        """Read data from an h5 File or group f."""
        if self._h5_data_keys is None:
            keys = self._default_data_keys()
        else:
            keys = self._h5_data_keys
        self._read_data(f, keys)
        self.h5_prepare_subs()
        self._read_subordinates(f)

    def _read_data(self, f, keys):
        d = _read_attrs(f)
        unread_keys = set(keys) - set(d.keys())
        if len(unread_keys) > 0:
            raise Exception("Could not read keys: %s"%(unread_keys))
        self.__dict__.update(d)

    def _read_subordinates(self, f):
        for k in self._h5_subordinate_keys:
            try:
                getattr(self, k)._read_h5(f[k])
            except Exception as e:
                raise Exception("%s could not be read: %s"%(k, e))


class H5ObjectList(SimpleH5Object):
    """
    A variable length list of SaveH5Objects.
    Note that a SimpleH5Object containing an H5ObjectList as an attribute
    must initialize a new H5ObjectList with the correct class instances
    in its h5_prepare_subs method.
    """

    def __init__(self, object_list):
        """object_list should be a list of SimpleH5Objects"""
        super(H5ObjectList, self).__init__(data_keys=[])
        self.object_list = object_list

    def _write_h5(self, f):
        """Writes each object in the object_list to a group"""
        for i, item in enumerate(self.object_list):
            g = f.create_group(_list_item_string(i))
            self.object_list[i]._write_h5(g)

    def _read_h5(self, f):
        """Loads each object in the object_list from the h5 groups"""
        for i in range(len(self.object_list)):
            g = f[_list_item_string(i)]
            self.object_list[i]._read_h5(g)

    def __getitem__(self, item):
        return self.object_list[item]

    def __len__(self):
        return len(self.object_list)

    def __iter__(self):
        for item in self.object_list:
            yield item

    def append(self, item):
        self.object_list.append(item)


class H5ObjectDict(SimpleH5Object):
    """
    A dictionary where each value is a SaveH5Object.
    Note that another SaveH5Object containing an H5ObjectDict as an attribute
    must initialize a new H5ObjectDict with the correct keys and the correct
    class instances for the values in its h5_prepare_subs method.
    """

    def __init__(self, object_dict):
        """object_dict should be a dict with SimpleH5Objects as values"""
        super(H5ObjectDict, self).__init__(data_keys=[])
        self.object_dict = object_dict

    def _write_h5(self, f):
        """Writes each k, v pair as a group"""
        keys = []
        for i, (k, v) in enumerate(self.object_dict.items()):   # inefficient in py2
            keys.append(k)
            g = f.create_group(_list_item_string(i))
            v._write_h5(g)
        _write_attr(f, OBJ_DICT_KEY_STR, keys)

    def _read_h5(self, f):
        """Loads each object in the object_dict from the h5 groups"""
        d = _read_attrs(f)
        keys = d[OBJ_DICT_KEY_STR]
        for k, v in self.object_dict.items():   # inefficient in py2
            idx = keys.index(k)
            v._read_h5(f[_list_item_string(idx)])

    def __getitem__(self, k):
        return self.object_dict[k]

    def __len__(self):
        return len(self.object_dict)

    def iteritems(self):
        for k, v in list(self.object_dict.items()):   # inefficient in py2
            yield k, v
