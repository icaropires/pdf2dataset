import io
import os

import numpy as np
import pytesseract
import cv2
import pdftotext
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from PIL import Image as PilImage

from .extract_task import ExtractTask, feature


class Image:

    def __init__(self, image, image_format=None):
        self.pil_image = image
        self.image_format = image_format or self.pil_image.format

    @classmethod
    def from_bytes(cls, image_bytes):
        image = PilImage.open(io.BytesIO(image_bytes))
        return cls(image)

    def resize(self, size):
        pil_image = self.pil_image.resize(size)
        return type(self)(pil_image, self.image_format)

    def preprocess(self):
        image = np.asarray(self.pil_image.convert('L'))
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

        pil_image = PilImage.fromarray(image)
        return type(self)(pil_image)

    @staticmethod
    def parse_size(image_size_str):
        if image_size_str is None:
            return None

        image_size_str = image_size_str.lower()

        try:
            width, height = map(int, image_size_str.split('x'))
        except ValueError as e:
            raise ValueError(
                f'Invalid image size parameter: {image_size_str}'
            ) from e

        return width, height

    def ocr(self, lang='por'):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(self.pil_image, lang=lang)

    def to_bytes(self):
        image_stream = io.BytesIO()

        with io.BytesIO() as image_stream:
            self.pil_image.save(image_stream, self.image_format)
            image_bytes = image_stream.getvalue()

        return image_bytes


class PdfExtractTask(ExtractTask):

    class OcrError(Exception):
        ...

    fixed_featues = ('path', 'page')

    def __init__(self, path, page, *args,
                 ocr=False, ocr_image_size=None, ocr_lang='por',
                 image_format='jpeg', image_size=None, **kwargs):

        self.page = page
        self.ocr = ocr
        self.ocr_lang = ocr_lang
        self.ocr_image_size = ocr_image_size
        self.image_format = image_format
        self.image_size = Image.parse_size(image_size)

        super().__init__(path, *args, **kwargs)

    def _extract_text_ocr(self):
        image_bytes, _ = self.get_feature('image_original')

        if not image_bytes:
            raise self.OcrError("Wasn't possible to get page image!")

        image = Image.from_bytes(image_bytes)
        preprocessed = image.preprocess()

        return preprocessed.ocr()

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

        image = Image(images[0])

        return image.to_bytes()

    @feature('int16')
    def get_page(self):
        return self.page

    @feature('string')
    def get_path(self):
        return str(self.path)

    @feature('binary')
    def get_image(self):
        image_bytes, _ = self.get_feature('image_original')

        if not image_bytes:
            return None

        if self.image_size:
            image = Image.from_bytes(image_bytes)
            size = self.image_size
            image_bytes = image.resize(size).to_bytes()

        return image_bytes

    @feature('string', exceptions=[pdftotext.Error, OcrError])
    def get_text(self):
        if self.ocr:
            return self._extract_text_ocr()

        return self._extract_text_native()
