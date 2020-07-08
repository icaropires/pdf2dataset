import os
import traceback

import numpy as np
import pytesseract
import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError


class ExtractionTask:

    def __init__(self, doc, page, lang='por'):
        self.doc = doc
        self.page = page
        self.lang = lang

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
        img = convert_from_path(
            self.doc,
            first_page=self.page,
            single_file=True,
            size=(None, 1100),
            fmt='jpeg'
        )

        return img[0]

    def process(self):
        text, error = None, None

        # Ray can handle the exceptions, but this makes switching to
        #   multiprocessing easy
        try:
            img = self.get_page_img()

            # TODO: Use OCR?
            text = self.ocr_image(img)

        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        return text, error
