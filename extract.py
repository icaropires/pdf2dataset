#!/bin/env python3

import argparse
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from ray.util.multiprocessing import Pool
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
        text, error = None, None

        try:
            img = self.get_page_img()
            text = self.ocr_image(img)
        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        return text, error


class TextExtraction:
    default_workers = min(32, os.cpu_count() + 4)  # Python 3.8 default

    def __init__(self, input_dir, output_dir, *, max_workers=default_workers):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers

    @staticmethod
    def _list_pages(d):
        try:
            num_pages = pdfinfo_from_path(d)['Pages']
            pages = range(1, num_pages+1)
        except PDFPageCountError:
            pages = [-1]  # Handled when processing

        return pages

    def gen_tasks(self, docs):
        '''
        Returns tasks to be processed.
        For faulty documents, only the page -1 will be available
        '''
        # 10 is not a mistake, is because this is a fast operation
        chunksize = max(1, (len(docs)/self.max_workers)//10)

        print()
        tasks = []
        with tqdm(desc='Generating tasks', unit='tasks') as pbar:
            with Pool() as pool:
                results = pool.imap(
                    self._list_pages, docs, chunksize=chunksize
                )
                results = zip(docs, results)

                for doc, range_pages in results:
                    new_tasks = [ExtractionTask(doc, p) for p in range_pages]
                    tasks += new_tasks
                    pbar.update(len(new_tasks))

        return tasks

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

    def save_result(self, result, task, error):
        if error is not None:
            page = task.page if task.page != -1 else 'doc'
            error_suffix = f'_{page}_error.log'

            output_path = self.get_output_path(task.doc)
            name = output_path.stem + error_suffix

            error_file = output_path.with_name(name)
            error_file.write_text(error)

        elif result is not None:
            output_path = self.get_output_path(task.doc)
            name = f'{output_path.stem}_{task.page}.txt'

            results_file = output_path.with_name(name)
            results_file.write_text(result)

    @staticmethod
    def _process_task(task):
        result, error = task.process()
        return task, result, error

    def apply(self):
        pdf_files = self.get_docs(input_dir)
        tasks = self.gen_tasks(pdf_files)

        chunksize = max(1, (len(tasks)/self.max_workers)//100)

        print(f'\nPDFs directory: {self.input_dir}',
              f'Output directory: {self.output_dir}',
              f'Using {self.max_workers} workers',
              f'Chunksize: {chunksize}', sep='\n', end='\n\n')

        def get_bar(results):
            return tqdm(
                results, total=len(tasks), desc='Processing pages',
                unit='pages', dynamic_ncols=True
            )

        with Pool() as pool:
            results = pool.imap_unordered(
                self._process_task, tasks, chunksize=chunksize
            )

            for task, result, error in get_bar(results):
                self.save_result(result, task, error)  # TODO: threads, async


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
