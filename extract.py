#!/bin/env python3

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pdf2image import (
    convert_from_path,
    pdfinfo_from_path,
)
from pdf2image.exceptions import PDFPageCountError
from tqdm import tqdm


# TODO: Add typing
# TODO: Return everything joined


class ExtractionTask:

    def __init__(self, doc, page):
        self.doc = doc
        self.page = page

    @staticmethod
    def ocr_image(img):
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

        return pytesseract.image_to_string(erd, lang='por')

    def get_page_img(self):
        img = convert_from_path(
            self.doc,
            first_page=self.page,
            single_file=True,
            fmt='jpeg'
        )

        return img[0]

    def process(self):
        img = self.get_page_img()
        text = self.ocr_image(img)

        return text


class TextExtraction:
    default_workers = min(32, os.cpu_count() + 4)  # Python 3.8 default

    def __init__(self, input_dir, output_dir, *, max_workers=default_workers):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers

    @staticmethod
    def generate_tasks(docs):
        '''
        Returns tasks to be processed.
        For faulty documents, only the page -1 will be available
        '''
        def list_pages(d):
            try:
                num_pages = pdfinfo_from_path(d)['Pages']
                pages = range(1, num_pages+1)
            except PDFPageCountError:
                pages = [-1]  # Handled when processing

            return pages

        return (ExtractionTask(d, p) for d in docs for p in list_pages(d))

    @staticmethod
    def get_docs(input_dir):
        pdf_files = input_dir.rglob('*.pdf')

        # Here feedback is better than keep using the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='docs'))

    def get_output_path(self, doc_path):
        relative = doc_path.relative_to(self.input_dir)
        out_path = self.output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Side effect

        return out_path

    def _get_result(self, task, future):
        result = None

        try:
            result = future.result()
        except Exception as e:
            page = task.page if task.page != -1 else 'doc'
            error_suffix = f'_{page}_error.log'

            output_path = self.get_output_path(task.doc)
            name = output_path.stem + error_suffix

            error_file = output_path.with_name(name)
            error_file.write_text(str(e))

        return result

    def save_result(self, result, task):
        if result is not None:
            output_path = self.get_output_path(task.doc)
            name = f'{output_path.stem}_{task.page}.txt'

            results_file = output_path.with_name(name)
            results_file.write_text(result)

    def apply(self):
        pdf_files = self.get_docs(input_dir)
        tasks = list(self.generate_tasks(pdf_files))

        print(f'\nPDFs directory: {self.input_dir}'
              f'\nOutput directory: {self.output_dir}'
              f'\nUsing {self.max_workers} workers', end='\n\n')

        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            with tqdm(total=len(tasks), desc='Processing pages',
                      unit='pages', dynamic_ncols=True) as pbar:
                def submit(task):
                    future = ex.submit(task.process)
                    future.add_done_callback(lambda _: pbar.update())

                    return future

                fs_to_tasks = {submit(t): t for t in tasks}
                for f in as_completed(fs_to_tasks):
                    task = fs_to_tasks[f]
                    result = self._get_result(task, f)
                    self.save_result(result, task)  # TODO: parallelize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract text from all PDF files in a directory'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='The folder to lookup for PDF files recursively'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help=('The folder to keep all the results, including log files and'
              ' intermediate files')
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=TextExtraction.default_workers,
        help='Workers to use in the pool'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    max_workers = args.workers

    extraction = TextExtraction(input_dir, output_dir, max_workers=max_workers)
    extraction.apply()
