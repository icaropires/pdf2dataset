#!/bin/env python3

import tempfile
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

import pandas as pd
import fastparquet
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
import pytesseract
import ray
from ray.util.multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path


# TODO: Add typing


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


class TextExtraction:
    def __init__(
        self, input_dir, results_file, *,
        tmp_dir=None, lang='por', **kwargs
    ):
        self.input_dir = Path(input_dir).resolve()
        self.results_file = Path(results_file).resolve()

        assert self.input_dir.is_dir()

        if self.results_file.exists():
            logging.warn(f'{results_file} already exists!'
                         ' Results will be appended to it!')

        if tmp_dir:
            self.tmp_dir = Path(tmp_dir).resolve()
        else:
            self.tmp_dir = Path(tempfile.mkdtemp())

        self.lang = lang

        self._df_lock = threading.Lock()
        self._chunk_df_size = 10000  # Dask default

        ray.init(ignore_reinit_error=True, **kwargs)
        print()  # Shame

        self.num_cpus = kwargs.get('num_cpus', None)
        self.num_cpus = self.num_cpus or os.cpu_count()

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
        # 10 because this is a fast operation
        chunksize = max(1, (len(docs)/self.num_cpus)//10)

        tasks = []
        with tqdm(desc='Generating tasks', unit='tasks') as pbar:
            with Pool() as pool:
                results = pool.imap(
                    self._list_pages, docs, chunksize=chunksize
                )

                for doc, range_pages in zip(docs, results):
                    if isinstance(range_pages, Exception):
                        raise range_pages

                    new_tasks = [ExtractionTask(doc, p, self.lang)
                                 for p in range_pages]
                    tasks += new_tasks
                    pbar.update(len(new_tasks))

        return tasks

    @staticmethod
    def get_docs(input_dir):
        pdf_files = input_dir.rglob('*.pdf')

        # Here feedback is better than keeping use of the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='docs'))

    def get_output_path(self, doc_path):
        relative = doc_path.relative_to(self.input_dir)
        out_path = self.tmp_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Side effect

        return out_path

    def _get_savinginfo(self, task_result):
        '''
        Saves the temporary results for each file and returns the path
        '''
        task, result, error = task_result
        output_path = self.get_output_path(task.doc)

        text = None
        if result is not None:
            name = f'{output_path.stem}_{task.page}.txt'
            text = result
            is_error = False

        elif error is not None:
            page = task.page if task.page != -1 else 'doc'
            error_suffix = f'_{page}_error.log'

            name = output_path.stem + error_suffix
            text = error
            is_error = True

        else:
            raise RuntimeError("Processing failed and no errors detected!")

        tmpfile = output_path.with_name(name)
        return tmpfile, text, is_error

    def _append_df(self, texts, errors):
        df = pd.DataFrame()

        if texts:
            path, texts = zip(*texts)
            df = pd.DataFrame({'path': path, 'text': texts, 'error': ''},
                              dtype='str')

        if errors:
            path, errors = zip(*errors)

            df = pd.concat([
                df,
                pd.DataFrame(
                    {'path': path, 'text': '', 'error': errors}, dtype='str'
                ),
            ])

        with self._df_lock:
            fastparquet.write(
                str(self.results_file), df,
                file_scheme='hive', compression='gzip',
                append=self.results_file.exists()
            )

        return df

    @staticmethod
    def _process_task(task):
        result, error = task.process()
        return task, result, error

    def apply(self):
        pdf_files = self.get_docs(self.input_dir)
        tasks = self.gen_tasks(pdf_files)

        chunksize = max(1, (len(tasks)/self.num_cpus)//100)

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.results_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {chunksize} (calculated)',
              f'Temporary directory: {self.tmp_dir}',
              sep='\n', end='\n\n')

        def get_bar(results):
            return tqdm(
                results, total=len(tasks), desc='Processing pages',
                unit='pages', dynamic_ncols=True
            )

        # There can be many files to be saved, so, doing async
        with ThreadPoolExecutor() as thread_exe:
            thread_fs, texts, errors = [], [], []

            # TODO: skip already preprocessed on tmp
            with Pool() as pool:  # May be distributed
                tasks_results = pool.imap_unordered(
                    self._process_task, tasks, chunksize=chunksize
                )

                for task_result in get_bar(tasks_results):

                    if isinstance(task_result, Exception):
                        raise task_result

                    path, text, is_error = self._get_savinginfo(task_result)

                    if not is_error:
                        texts.append((path, text))
                    else:
                        errors.append((path, text))

                    thread_fs.append(
                        thread_exe.submit(path.write_text, text)
                    )

                    if len(texts) + len(errors) >= self._chunk_df_size:
                        # Persist to disk, aiming large amount of data
                        thread_fs.append(
                            thread_exe.submit(self._append_df, texts, errors)
                        )
                        texts, errors = [], []

            if texts or errors:
                thread_fs.append(
                    thread_exe.submit(self._append_df, texts, errors)
                )

            for f in thread_fs:  # Avoid fail silently
                f.result()
