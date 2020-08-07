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
import base64


class ExtractionTask:

    def __init__(self, doc, page, doc_bin=None, *, lang='por',
                 ocr=False, img_column=False, img_size=None):
        self.doc = doc
        self.doc_bin = doc_bin
        self.page = page
        self.lang = lang
        self.ocr = ocr
        self.img_column = img_column
        self.img_size = img_size

    def load_bin(self, enforce=False):
        '''
        Loads the document binary

        Should not be called inside the same class, as the node running this
        task might not have access to the document in his filesystem
        '''
        if enforce or not self.doc_bin:
            self.doc_bin = self.doc.read_bytes()

    def copy(self):
        return copy.copy(self)

    def preprocess_image(self, img):
        tsh = np.array(img.convert('L'))
        tsh = cv2.adaptiveThreshold(
            tsh, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            97, 50
        )

        erd = cv2.erode(
            tsh,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
            iterations=1
        )

        return erd

    def ocr_image(self, img):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(img, lang=self.lang)

    def encode_image(self, img):
        if self.img_size:
            img_size = (int(self.img_size.split('x')[0]),
                        int(self.img_size.split('x')[1]))
            img = cv2.resize(img, img_size)
        img_encoded = cv2.imencode('.jpg', img)[1].tostring()
        img_as_b64 = base64.b64encode(img_encoded)
        return img_as_b64

    def get_page_img(self):
        img = convert_from_bytes(
            self.doc_bin,
            first_page=self.page,
            single_file=True,
            size=(None, 1100),
            fmt='jpeg'
        )

        return img[0]

    def _process_ocr(self):
        text, img_encoded, img_preprocessed, error = None, None, None, None

        try:
            img = self.get_page_img()
            img_preprocessed = self.preprocess_image(img)
            text = self.ocr_image(img_preprocessed)
            if self.img_column:
                img_encoded = self.encode_image(img_preprocessed)
        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        if self.img_column:
            return (text, img_encoded), error
        else:
            return text, error

    def _process_native(self):
        text, img_encoded, img_preprocessed, error = None, None, None, None

        try:
            with io.BytesIO(self.doc_bin) as f:
                pages = pdftotext.PDF(f)
                text = pages[self.page-1]
            if self.img_column:
                img = self.get_page_img()
                img_preprocessed = self.preprocess_image(img)
                img_encoded = self.encode_image(img_preprocessed)
        except pdftotext.Error:
            error = traceback.format_exc()

        if self.img_column:
            return (text, img_encoded), error
        else:
            return text, error

    def process(self):
        if not self.doc_bin:
            raise RuntimeError(
                "'doc_bin' can't be empty for processing the task"
            )

        if self.ocr:
            return self._process_ocr()

        return self._process_native()
