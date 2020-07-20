#!/bin/env python3

import os
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import itertools as it

import pandas as pd
import fastparquet
import ray
from ray.util.multiprocessing import Pool
from ray.util.multiprocessing.pool import PoolTaskError
from tqdm import tqdm
from pathlib import Path
import pdftotext

from .extraction_task import ExtractionTask


# TODO: Add typing
# TODO: Move some methods other class/file
# TODO: Set up a logger to the class
# TODO: Substitute most (all?) prints for logs
# TODO: Create a task result namedtuple


class TextExtraction:
    _path_pat = r'((?P<path>.+)_(?P<page>(-?\d+|doc))(.txt|_error.log))'

    def __init__(
        self, input_dir, results_file, *,
        tmp_dir='', lang='por', ocr=False, **kwargs
    ):
        self.input_dir = Path(input_dir).resolve()
        self.results_file = Path(results_file).resolve()
        assert self.input_dir.is_dir()

        # Keep str and not Path, custom behaviour if is empty
        self.tmp_dir = tmp_dir

        self.lang = lang
        self.ocr = ocr

        self._df_lock = threading.Lock()
        self._chunk_df_size = 10000  # Dask default

        ray.init(ignore_reinit_error=True, **kwargs)
        print()  # Shame

        self.num_cpus = kwargs.get('num_cpus', None)
        self.num_cpus = self.num_cpus or os.cpu_count()

        # Warning after the ray.init logging
        if self.results_file.exists():
            logging.warning(f'{results_file} already exists!'
                            ' Results will be appended to it!')

    @staticmethod
    def get_docs(input_dir):
        pdf_files = input_dir.rglob('*.pdf')

        # Here feedback is better than keeping use of the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='docs'))

    def _get_output_path(self, doc_path):
        relative = doc_path.relative_to(self.input_dir)
        out_path = Path(self.tmp_dir) / relative

        if self.tmp_dir:
            out_path.parent.mkdir(parents=True, exist_ok=True)  # Side effect

        return out_path

    def _get_savingpath(self, task, is_error=False):
        output_path = self._get_output_path(task.doc)
        name = f'{output_path.stem}_{task.page}.txt'

        if is_error:
            # Commented to keep int16 instead of str
            # page = task.page if task.page != -1 else 'doc'
            error_suffix = f'_{task.page}_error.log'
            name = output_path.stem + error_suffix

        return output_path.with_name(name)

    def _get_savinginfo(self, task_result):
        '''
        Saves the temporary results for each file and returns the path
        '''
        task, result, error = task_result

        text = None
        if result is not None:
            text = result
            is_error = False

        elif error is not None:
            text = error
            is_error = True

        else:
            raise RuntimeError(
                "Processing failed and no errors were detected!"
            )

        tmp_file = self._get_savingpath(task, is_error)
        return tmp_file, text, is_error

    @staticmethod
    def _preprocess_path(df):  # Side-effect
        parsed = df['path'].str.extract(TextExtraction._path_pat)

        df['page'] = parsed['page'].astype('int16')
        df['path'] = parsed['path'] + '.pdf'

        return df

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

        df = self._preprocess_path(df)
        df = df[['path', 'page', 'text', 'error']]

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

    @staticmethod
    def _list_pages(doc):
        # Using pdftotext to get num_pages because it's the best way I know
        # pdftotext extracts lazy, so this won't process the text

        try:
            with doc.open('rb') as f:
                num_pages = len(pdftotext.PDF(f))

            pages = range(1, num_pages+1)
        except pdftotext.Error:
            pages = [-1]

        return pages

    def _gen_tasks(self, docs):
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
                    if isinstance(range_pages, PoolTaskError):
                        raise range_pages.underlying

                    new_tasks = [
                        ExtractionTask(doc, doc.read_bytes(), p,
                                       self.lang, self.ocr)
                        for p in range_pages
                    ]
                    tasks += new_tasks
                    pbar.update(len(new_tasks))

        return tasks

    def _split_processed_tasks(self, tasks):
        def get_taskinfo(task):
            for is_error in (True, False):
                filename = self._get_savingpath(task, is_error)

                if filename.exists():
                    text = filename.read_text()
                    if is_error:
                        return (task, None, text)  # TODO: Task result

                    return (task, text, None)  # TODO: Task result

            return None

        processed, not_processed = [], []
        for task in tasks:
            tasks_result = get_taskinfo(task)

            if tasks_result:
                processed.append(tasks_result)
            else:
                not_processed.append(task)

        return processed, not_processed

    def _get_tasks(self, docs):
        tasks = self._gen_tasks(docs)
        processed, not_processed = self._split_processed_tasks(tasks)

        if len(processed):
            logging.warning(
                f"Skipping {len(processed)} already"
                f" processed pages in directory '{self.tmp_dir}'"
            )

        return processed, not_processed

    def apply(self):
        pdf_files = self.get_docs(self.input_dir)

        tasks = self._get_tasks(pdf_files)
        chunksize = max(1, (len(tasks)/self.num_cpus)//100)

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.results_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {chunksize} (calculated)',
              f'Temporary directory: {self.tmp_dir or "not used"}',
              sep='\n', end='\n\n')

        # There can be many files to be saved, so, doing async
        with ThreadPoolExecutor() as thread_exe:
            thread_fs, texts, errors = [], [], []
            processed, not_processed = tasks

            with Pool() as pool:  # May be distributed
                processing_tasks = pool.imap_unordered(
                    self._process_task, not_processed, chunksize=chunksize
                )
                tasks_results = it.chain(processed, processing_tasks)

                for task_result in tqdm(
                    tasks_results, total=len(processed) + len(not_processed),
                    desc='Processing pages', unit='pages', dynamic_ncols=True
                ):
                    if isinstance(task_result, PoolTaskError):
                        raise task_result.underlying

                    path, text, is_error = self._get_savinginfo(task_result)

                    if not is_error:
                        texts.append((path, text))
                    else:
                        errors.append((path, text))

                    if self.tmp_dir:
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
