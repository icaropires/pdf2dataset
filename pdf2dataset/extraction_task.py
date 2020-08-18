import io
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

    def __init__(self, doc, page, doc_bin=None, *, lang='por',
                 ocr=False, img=False, img_size=None):
        self.doc = doc
        self.doc_bin = doc_bin
        self.page = page
        self.lang = lang
        self.ocr = ocr
        self.img = img
        self.img_size = img_size

        self._features = {'doc': str(self.doc), 'page': self.page}
        self._errors = {'doc': None, 'page': None}

        # TODO: loop
        self._features['text'] = None
        self._errors['text'] = None

        if self.img:
            self._features['image'] = None
            self._errors['image'] = None

        self._tmp_features = {'image_original': 'image'}
        self._init_tmp_features()
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

    def _init_tmp_features(self):
        for tmp in self._tmp_features:
            self._features[tmp] = None
            self._errors[tmp] = None

    def _pop_tmp_features(self):
        keys = list(self._tmp_features)

        for tmp in keys:
            self._features.pop(tmp)  # Changing dict size

    def _gen_tmpfeatures_methods(self):
        # Can't be lambda, not pickle serializable
        def get_matching(matching):
            return getattr(self, self._get_extractmethod(matching))

        for tmp, matching in self._tmp_features.items():
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

    @staticmethod
    def _img_to_bytes(img):
        img_stream = io.BytesIO()
        img.save(img_stream, 'jpeg')

        return img_stream.getvalue()

    def extract_image(self):
        img, error = None, None

        try:
            imgs = convert_from_bytes(
                self.doc_bin,
                first_page=self.page, single_file=True, fmt='jpeg'
            )
            img = imgs[0]

            self._features['image_original'] = self._img_to_bytes(img)

            if self.img:
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

        # TODO: make it meta!
        self._features['text'], _ = self._get_feature('text')

        if self.img:
            self._features['image'], _ = self._get_feature('image')

        error = '\n\n\n\n'.join(
            f'{f}:\n{e}' for f, e in self._errors.items() if e
        )
        error = error or None

        self._pop_tmp_features()
        return {**self._features, 'error': error}
