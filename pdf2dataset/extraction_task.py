import io
import itertools as it
import os
import traceback
import copy

import numpy as np
import pytesseract
import cv2
import pdftotext
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from PIL import Image


class ExtractionTask:

    _extractmethods_prefix = 'extract_'
    _fixed_featues = ['doc', 'page']

    # Refers to some real feature but will never be returned as result,
    # useful to methods use its value
    _tmp_features_redirect = {
        'image_original': 'image'
    }

    _img_format = 'jpeg'

    def __init__(self, doc, page, doc_bin=None, *, lang='por',
                 ocr=False, features='text', img_size=None):
        self.doc = doc
        self.doc_bin = doc_bin
        self.page = page
        self.lang = lang
        self.ocr = ocr
        self.img_size = img_size
        self._features = {}
        self._errors = {}

        self._init_all_features(features)
        self._gen_tmpfeatures_methods()

    def load_bin(self, enforce=False):
        '''
        Loads the document binary

        Should not be called inside the same class, as the node running this
        task might not have access to the document in his filesystem
        '''
        if enforce or not self.doc_bin:
            self.doc_bin = self.doc.read_bytes()

    def copy(self):
        return copy.deepcopy(self)

    def is_feature_selected(self, feature):
        return feature in self._features

    def _init_all_features(self, features):
        features = it.chain(self._fixed_featues, features,
                            self._tmp_features_redirect)

        self._features = {f: None for f in features}
        self._errors = copy.deepcopy(self._features)

    def _pop_tmp_features(self):
        keys = list(self._tmp_features_redirect)

        for tmp in keys:
            self._features.pop(tmp)  # Changing dict size

    def _gen_tmpfeatures_methods(self):
        # Can't be lambda, not pickle serializable
        def get_matching(matching):
            return getattr(self, self._get_extractmethod(matching))

        for tmp, matching in self._tmp_features_redirect.items():
            setattr(self, self._get_extractmethod(tmp), get_matching(matching))

    @classmethod
    def _get_extractmethod(cls, feature):
        return cls._extractmethods_prefix + feature

    def _get_feature(self, feature):
        extract_method_name = self._get_extractmethod(feature)
        extract_method = getattr(self, extract_method_name)

        if self._features[feature] is None and self._errors[feature] is None:
            self._features[feature], self._errors[feature] = extract_method()

        return self._features[feature], self._errors[feature]

    def _check_result_fixedfeatures(self):
        for fixed in self._fixed_featues:
            error_msg = f'Missing {fixed} in results'
            assert fixed in self._features, error_msg

    def _preprocess_img(self, img):
        img = np.array(img.convert('L'))
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            97, 50
        )

        img = cv2.erode(
            img,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
            iterations=1
        )

        return img

    def _ocr_img(self, img):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(img, lang=self.lang)

    def _extract_text_ocr(self):
        img_bytes, error = self._get_feature('image_original')

        if error:
            return None, error

        img = Image.open(io.BytesIO(img_bytes))

        img_preprocessed = self._preprocess_img(img)
        text = self._ocr_img(img_preprocessed)

        img.close()
        return text, None

    def _extract_text_native(self):
        text, error = None, None

        try:
            with io.BytesIO(self.doc_bin) as f:
                pages = pdftotext.PDF(f)
                text = pages[self.page-1]
        except pdftotext.Error:
            error = traceback.format_exc()

        return text, error

    @classmethod
    def _img_to_bytes(cls, img):
        img_stream = io.BytesIO()
        img.save(img_stream, cls._img_format)

        return img_stream.getvalue()

    def extract_page(self):
        return self.page, None

    def extract_doc(self):
        return str(self.doc), None

    def extract_image(self):
        img, error = None, None

        try:
            imgs = convert_from_bytes(
                self.doc_bin, first_page=self.page,
                single_file=True, fmt=self._img_format
            )
            img = imgs[0]

            self._features['image_original'] = self._img_to_bytes(img)

            if self.is_feature_selected('image'):
                self._features['image'] = self._features['image_original']

            if self.img_size:
                img_size = tuple(int(x) for x in self.img_size.split('x'))
                self._features['image'] = img.resize(img_size)

        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        return self._features['image_original'], error

    def extract_text(self):
        if self.ocr:
            return self._extract_text_ocr()

        return self._extract_text_native()

    def process(self):
        if not self.doc_bin:
            raise RuntimeError(
                "'doc_bin' can't be empty for processing the task"
            )

        for feature in self._features:
            self._features[feature], _ = self._get_feature(feature)

        # TODO: improve
        error = '\n\n\n\n'.join(
            f'{f}:\n{e}' for f, e in self._errors.items() if e
        )
        error = error or None

        self._pop_tmp_features()
        self._check_result_fixedfeatures()

        return {**self._features, 'error': error}
