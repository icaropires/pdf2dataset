import io
import os
import traceback

import numpy as np
import pytesseract
import cv2
import pdftotext
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError


class ExtractionTask:

    def __init__(self, doc, doc_bin, page, lang='por', ocr=False):
        self.doc = doc
        self.doc_bin = doc_bin
        self.page = page
        self.lang = lang
        self.ocr = ocr

    def ocr_image(self, img):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'

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

        return pytesseract.image_to_string(erd, lang=self.lang)

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
        text, error = None, None

        try:
            img = self.get_page_img()
            text = self.ocr_image(img)
        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        return text, error

    def _process_native(self):
        text, error = None, None

        try:
            with io.BytesIO(self.doc_bin) as f:
                text = pdftotext.PDF(f)[self.page-1]
        except pdftotext.Error:
            error = traceback.format_exc()

        return text, error

    def process(self):
        if self.ocr:
            return self._process_ocr()

        return self._process_native()
