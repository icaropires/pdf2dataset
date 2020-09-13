import traceback
from abc import ABC
from copy import deepcopy
from itertools import chain
from inspect import getmembers, isroutine
from functools import wraps

import pyarrow as pa


def feature(pyarrow_type=None, is_helper=False, exceptions=None, **type_args):
    exceptions = exceptions or tuple()
    exceptions = tuple(exceptions)

    if not (pyarrow_type or is_helper):
        raise ValueError("If feature is not helper, must set 'pyarrow_type'")

    def decorator(feature_method):
        feature_method.pyarrow_type = None
        feature_method.is_feature = True
        feature_method.is_helper = is_helper

        if pyarrow_type is not None:
            type_ = getattr(pa, pyarrow_type)(**type_args)

            if isinstance(type_, pa.DataType):
                feature_method.pyarrow_type = type_
            else:
                raise ValueError(f'Invalid PyArrow type {pyarrow_type}!')

        @wraps(feature_method)
        def inner(*args, **kwargs):
            result, error = None, None

            try:
                result = feature_method(*args, **kwargs)
            except exceptions:
                error = traceback.format_exc()

            return result, error
        return inner

    return decorator


# TODO: Eventually, I'll make this a new lib
class ExtractTask(ABC):

    fixed_featues = ('path',)
    _feature_prefix = 'get_'  # Optional

    def __init__(self, path, file_bin=None, sel_features='all'):
        self.path = path
        self.file_bin = file_bin
        self.sel_features = self._parse_sel_features(sel_features)

        self._features = {}
        self._errors = {}

        self._init_all_features()

    def __init_subclass__(cls, **kwargs):
        # Memoization
        cls._helper_list = None
        cls._features_list = {}

    @classmethod
    def list_helper_features(cls):
        if cls._helper_list is not None:
            return cls._helper_list.copy()

        prefix = cls._feature_prefix

        def is_helper(name, method):
            return (getattr(method, 'is_helper', False)
                    and name.startswith(prefix))

        class_routines = getmembers(cls, predicate=isroutine)

        cls._helper_list = [n[len(prefix):]
                            for n, m in class_routines if is_helper(n, m)]

        return cls._helper_list

    @classmethod
    def list_features(cls, *, exclude_fixed=False):
        is_calculated = cls._features_list.get(exclude_fixed)

        if is_calculated:
            return cls._features_list[exclude_fixed].copy()

        def include(name, method):
            helper = [cls._get_feature_methodname(f)
                      for f in cls.list_helper_features()]

            is_feature = getattr(method, 'is_feature', False)
            feat_name = name[len(cls._feature_prefix):]

            return (is_feature
                    and name not in helper
                    and name.startswith(cls._feature_prefix)
                    and not (feat_name in cls.fixed_featues and exclude_fixed))

        class_routines = getmembers(cls, predicate=isroutine)

        features_list = [n[len(cls._feature_prefix):]
                         for n, m in class_routines if include(n, m)]

        cls._features_list[exclude_fixed] = features_list
        return features_list

    @classmethod
    def get_schema(cls, features=()):
        def get_type(feature_name):
            method_name = cls._get_feature_methodname(feature_name)
            method = getattr(cls, method_name)

            if method.is_helper:
                return None

            return method.pyarrow_type

        class_features = cls.list_features()
        names = (name for name in features if name in class_features)

        features_types = ((name, get_type(name)) for name in names)

        features_types = [(f, t) for f, t in features_types if t is not None]
        features_types.append(('error', pa.string()))

        return pa.schema(features_types)

    @classmethod
    def _get_feature_methodname(cls, feature_name):
        method_name = cls._feature_prefix + feature_name

        if not hasattr(cls, method_name):
            raise RuntimeError(f"Method '{method_name}' not found!")

        return method_name

    def list_instance_features(self):
        return list(chain(self.fixed_featues, self.sel_features, ['error']))

    def load_bin(self, enforce=False):
        '''
        Loads the file binary

        Should not be called inside its class, as the node running
        this task might not have access to the file in his filesystem
        '''
        if enforce or not self.file_bin:
            self.file_bin = self.path.read_bytes()

    def copy(self):
        return deepcopy(self)

    def get_feature(self, name):
        extract_method_name = self._get_feature_methodname(name)
        extract_method = getattr(self, extract_method_name)

        if self._features[name] is None and self._errors[name] is None:
            self._features[name], self._errors[name] = extract_method()

        return self._features[name], self._errors[name]

    def process(self):
        if not self.file_bin:
            raise RuntimeError(
                "'file_bin' can't be empty for processing the task!"
            )

        return self._gen_result()

    def _gen_result(self):
        expected_features = chain(self.fixed_featues, self.sel_features)

        result = {name: self.get_feature(name)[0]
                  for name in expected_features}
        result['error'] = self._gen_errors_string()

        return result

    def _gen_errors_string(self):
        features_errors = (f'{f}:\n{e}' for f, e in self._errors.items() if e)
        all_errors = '\n\n\n'.join(features_errors)

        return all_errors or None

    def _init_all_features(self):
        helper = self.list_helper_features()
        features = chain(self.fixed_featues, helper, self.sel_features)

        self._features = {f: None for f in features}
        self._errors = deepcopy(self._features)

    def _parse_sel_features(self, sel_features):
        possible_features = self.list_features()

        if sel_features == '':
            sel_features = []

        elif sel_features == 'all':
            sel_features = possible_features

        elif isinstance(sel_features, list):
            ...

        else:
            sel_features = sel_features.split(',')

        failed = (f not in possible_features for f in sel_features)
        if any(failed):
            sel_features = ','.join(sel_features)
            possible_features = ','.join(possible_features)

            raise ValueError(
                f"Invalid feature list: '{sel_features}'"
                f"\nPossible features are: '{possible_features}'"
            )

        return sel_features
