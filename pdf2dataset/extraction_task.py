import io
import os
import traceback
from copy import deepcopy
from itertools import chain
from inspect import getmembers, isroutine
from functools import wraps

import numpy as np
import pytesseract
import cv2
import pdftotext
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from PIL import Image


class ExtractionTask:

    fixed_featues = ('path', 'page')

    _extractmethods_prefix = 'extract_'
    _helper_features = ('image_original',)

    _image_format = 'jpeg'

    def __init__(self, path, page, doc_bin=None, *, lang='por',
                 ocr=False, features='text', image_size=None):
        self.path = path
        self.doc_bin = doc_bin
        self.page = page
        self.lang = lang
        self.ocr = ocr
        self.image_size = image_size

        self._features = {}
        self._errors = {}

        self._init_all_features(features)

    @classmethod
    def _get_extractmethod(cls, feature):
        return cls._extractmethods_prefix + feature

    @classmethod
    def list_features(cls):
        prefix = cls._extractmethods_prefix

        class_routines = getmembers(cls, predicate=isroutine)
        extraction_methods = (
            n for n, _ in class_routines if n.startswith(prefix)
        )

        features = (n[len(prefix):] for n in extraction_methods)
        features = [n for n in features if n not in cls._helper_features]

        return features

    def load_bin(self, enforce=False):
        '''
        Loads the document binary

        Should not be called inside the same class, as the node running this
        task might not have access to the document in his filesystem
        '''
        if enforce or not self.doc_bin:
            self.doc_bin = self.path.read_bytes()

    def copy(self):
        return deepcopy(self)

    def is_feature_selected(self, feature):
        return feature in self._features

    def extraction_method(*exceptions):
        exceptions = exceptions or tuple()
        exceptions = tuple(exceptions)

        def decorator(extraction_method):
            @wraps(extraction_method)
            def inner(*args, **kwargs):
                result, error = None, None

                try:
                    result = extraction_method(*args, **kwargs)
                except exceptions:
                    error = traceback.format_exc()

                return result, error
            return inner

        return decorator

    def _init_all_features(self, features):
        features = chain(self.fixed_featues, self._helper_features, features)

        self._features = {f: None for f in features}
        self._errors = deepcopy(self._features)

    def _pop_helper_features(self):
        for helper in self._helper_features:
            self._features.pop(helper)

    def _get_feature(self, feature):
        extract_method_name = self._get_extractmethod(feature)
        extract_method = getattr(self, extract_method_name)

        if self._features[feature] is None and self._errors[feature] is None:
            self._features[feature], self._errors[feature] = extract_method()

        return self._features[feature], self._errors[feature]

    def _check_result_fixedfeatures(self):
        for fixed in self.fixed_featues:
            error_msg = f'Missing {fixed} in results'
            assert fixed in self._features, error_msg

    def _gen_errors_string(self):
        features_errors = (f'{f}:\n{e}' for f, e in self._errors.items() if e)
        all_errors = '\n\n\n'.join(features_errors)

        return all_errors or None

    def _ocr_image(self, image):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(image, lang=self.lang)

    def _preprocess_image(self, image):
        image = np.array(image.convert('L'))
        image = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            97, 50
        )

        image = cv2.erode(
            image,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
            iterations=1
        )

        return image

    @classmethod
    def _image_to_bytes(cls, image):
        image_stream = io.BytesIO()
        image.save(image_stream, cls._image_format)

        return image_stream.getvalue()

    def _extract_text_ocr(self):
        image_bytes, error = self._get_feature('image_original')

        if not image_bytes:
            return None

        image = Image.open(io.BytesIO(image_bytes))

        image_preprocessed = self._preprocess_image(image)
        text = self._ocr_image(image_preprocessed)

        image.close()
        return text

    def _extract_text_native(self):
        with io.BytesIO(self.doc_bin) as f:
            pages = pdftotext.PDF(f)
            text = pages[self.page-1]

        return text

    @extraction_method(PDFPageCountError, PDFSyntaxError)
    def extract_image_original(self):
        images = convert_from_bytes(
            self.doc_bin, first_page=self.page,
            single_file=True, fmt=self._image_format
        )

        image_bytes = self._image_to_bytes(images[0])

        return image_bytes

    @extraction_method()
    def extract_page(self):
        return self.page

    @extraction_method()
    def extract_path(self):
        return str(self.path)

    @extraction_method()
    def extract_image(self):
        image_bytes, error = self._get_feature('image_original')

        if not image_bytes:
            return None

        if self.image_size:
            image = Image.open(io.BytesIO(image_bytes))
            image_size = tuple(int(x) for x in self.image_size.split('x'))
            image_bytes = self._image_to_bytes(image.resize(image_size))

        return image_bytes

    @extraction_method(pdftotext.Error)
    def extract_text(self):
        if self.ocr:
            return self._extract_text_ocr()

        return self._extract_text_native()

    def process(self):
        if not self.doc_bin:
            raise RuntimeError(
                "'doc_bin' can't be empty for processing the task!"
            )

        for feature in self._features:
            self._features[feature], _ = self._get_feature(feature)

        self._pop_helper_features()
        self._check_result_fixedfeatures()

        return {**self._features, 'error': self._gen_errors_string()}
