import io
import os

import numpy as np
import pytesseract
import cv2
import pdftotext
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from PIL import Image

from .extract_task import ExtractTask, feature


class PdfExtractTask(ExtractTask):

    fixed_featues = ('path', 'page')

    def __init__(self, path, page, *args,
                 ocr=False, ocr_image_size=None, ocr_lang='por',
                 image_format='jpeg', image_size=None, **kwargs):

        self.page = page
        self.ocr = ocr
        self.ocr_lang = ocr_lang
        self.ocr_image_size = ocr_image_size
        self.image_format = image_format
        self.image_size = self._parse_image_size(image_size)

        super().__init__(path, *args, **kwargs)

    @staticmethod
    def _parse_image_size(image_size_str):
        if image_size_str is None:
            return None

        image_size_str = image_size_str.lower()

        try:
            width, height = map(int, image_size_str.split('x'))
        except ValueError:
            raise ValueError(f'Invalid image size parameter: {image_size_str}')

        return width, height

    def _ocr_image(self, image):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(image, lang=self.ocr_lang)

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

    def _image_to_bytes(self, image):
        image_stream = io.BytesIO()
        image.save(image_stream, self.image_format)

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
        with io.BytesIO(self.file_bin) as f:
            pages = pdftotext.PDF(f)
            text = pages[self.page-1]

        return text

    @feature('binary', is_helper=True,
             exceptions=(PDFPageCountError, PDFSyntaxError))
    def get_image_original(self):
        images = convert_from_bytes(
            self.file_bin, first_page=self.page,
            single_file=True, fmt=self.image_format,
            size=(None, self.ocr_image_size)
        )

        image_bytes = self._image_to_bytes(images[0])

        return image_bytes

    @feature('int16')
    def get_page(self):
        return self.page

    @feature('string')
    def get_path(self):
        return str(self.path)

    @feature('binary')
    def get_image(self):
        image_bytes, error = self._get_feature('image_original')

        if not image_bytes:
            return None

        if self.image_size:
            image = Image.open(io.BytesIO(image_bytes))
            size = self.image_size
            image_bytes = self._image_to_bytes(image.resize(size))

        return image_bytes

    @feature('string', exceptions=[pdftotext.Error])
    def get_text(self):
        if self.ocr:
            return self._extract_text_ocr()

        return self._extract_text_native()
