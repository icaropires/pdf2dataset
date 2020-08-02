#!/bin/env python3

import io
import itertools as it
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from more_itertools import ichunked

import pandas as pd
import fastparquet
import ray
from tqdm import tqdm
from pathlib import Path
import pdftotext

from .extraction_task import ExtractionTask


# TODO: Some day I will reduce this file size!!

# TODO: Add typing
# TODO: Set up a logger to the class
# TODO: Substitute most (all?) prints for logs
# TODO: Create a 'task result' namedtuple


class TextExtraction:
    _path_pat = r'((?P<path>.+)_(?P<page>(-?\d+|doc))(.txt|_error.log))'

    def __init__(
        self, input_dir, results_file='', *,
        tmp_dir='', lang='por', ocr=False, small=False,
        chunksize=None, chunk_df_size=10000, check_inputdir=True,
        max_docs_memory=3000, **ray_params
    ):

        self.input_dir = Path(input_dir).resolve()
        self.results_file = Path(results_file).resolve()

        if (check_inputdir and
                (not (self.input_dir.exists() and self.input_dir.is_dir()))):
            raise RuntimeError(f"Invalid input_dir: '{self.input_dir}',"
                               " it must exists and be a directory")

        if not small:
            if not results_file:
                raise RuntimeError("If not using 'small' arg,"
                                   " 'results_file' is mandatory")

            if self.results_file.exists():
                logging.warning(f'{results_file} already exists!'
                                ' Results will be appended to it!')

        # Keep str and not Path, custom behaviour if is empty string
        self.tmp_dir = tmp_dir

        self.num_cpus = ray_params.get('num_cpus') or os.cpu_count()
        self.ray_params = ray_params
        self.chunksize = chunksize
        self.small = small
        self.lang = lang
        self.ocr = ocr
        self.max_docs_memory = max_docs_memory

        self._df_lock = threading.Lock()
        self.chunk_df_size = chunk_df_size

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

    def _to_df(self, texts, errors):
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

        return df

    def _append_df(self, df):

        with self._df_lock:
            fastparquet.write(
                str(self.results_file), df,
                file_scheme='hive', compression='gzip',
                append=self.results_file.exists()
            )

    @staticmethod
    def _get_pages_range(doc, doc_bin=None):
        # Using pdftotext to get num_pages because it's the best way I know
        # pdftotext extracts lazy, so this won't process the text

        try:
            if not doc_bin:
                with doc.open('rb') as f:
                    num_pages = len(pdftotext.PDF(f))
            else:
                with io.BytesIO(doc_bin) as f:
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
        chunksize = int(max(1, (len(docs)/self.num_cpus)//10))

        tasks = []
        with Pool(self.num_cpus) as pool, \
                tqdm(desc='Counting pages', unit='pages') as pbar:

            results = pool.imap(
                self._get_pages_range, docs, chunksize=chunksize
            )

            for doc, range_pages in zip(docs, results):
                new_tasks = [
                    ExtractionTask(doc, p, lang=self.lang, ocr=self.ocr)
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

    @staticmethod
    def _get_apply_bar(tasks_results, num_tasks):
        return tqdm(
            tasks_results, total=num_tasks,
            desc='Processing pages', unit='pages', dynamic_ncols=True
        )

    @staticmethod
    def _load_task_bin(task):
        task = task.copy()  # Copy to have control over memory references
        task.load_bin()

        return task

    @staticmethod
    def process_task(task):
        result, error = task.process()
        return task, result, error

    @ray.remote
    def process_chunk_ray(chunk):
        return [TextExtraction.process_task(t) for t in chunk]

    def _ray_process(self, tasks):
        tasks = iter(tasks)
        futures = []

        chunks = ichunked(tasks, int(self.chunksize))

        for chunk in chunks:
            chunk = [self._load_task_bin(t) for t in chunk]
            futures.append(self.process_chunk_ray.remote(chunk))

            if len(futures) >= self.num_cpus + 4:
                break

        while futures:
            finished, rest = ray.wait(futures, num_returns=1)
            results = ray.get(finished[0])

            for result in results:
                yield result

            try:
                chunk = next(chunks)
                chunk = [self._load_task_bin(t) for t in chunk]
                rest.append(self.process_chunk_ray.remote(chunk))
            except StopIteration:
                ...

            futures = rest

    def _apply_big(self, tasks, num_tasks):
        'Apply the extractino to a big volume of data'

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.results_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {self.chunksize}',
              f'Temporary directory: {self.tmp_dir or "not used"}',
              sep='\n', end='\n\n')

        ray.init(**self.ray_params)

        # TODO: Add queue
        with ThreadPoolExecutor(max_workers=4) as thread_exec:
            thread_fs, texts, errors = [], [], []
            processed, not_processed = tasks

            not_processed = self._ray_process(not_processed)
            results = it.chain(processed, not_processed)

            for result in self._get_apply_bar(results, num_tasks):
                if isinstance(result, Exception):
                    raise result

                path, text, is_error = self._get_savinginfo(result)

                if not is_error:
                    texts.append((path, text))
                else:
                    errors.append((path, text))

                if self.tmp_dir:
                    thread_fs.append(
                        thread_exec.submit(path.write_text, text)
                    )

                if len(texts) + len(errors) >= self.chunk_df_size:
                    # Persist to disk, aiming large amount of data
                    df = self._to_df(texts, errors)

                    thread_fs.append(
                        thread_exec.submit(self._append_df, df)
                    )
                    texts, errors = [], []

            if texts or errors:
                df = self._to_df(texts, errors)

                thread_fs.append(
                    thread_exec.submit(self._append_df, df)
                )

            for f in thread_fs:  # Avoid fail silently
                f.result()

        ray.shutdown()

    def _apply_small(self, tasks, num_tasks):
        ''''Apply the extraction to a small volume of data
        More direct approach than 'big', but with these differences:
            - Not saving progress
            - Distributed processing not supported
            - Don't write dataframe to disk
            - Returns the resultant dataframe
        '''

        texts, errors = [], []
        processed, not_processed = tasks
        not_processed = (self._load_task_bin(t) for t in not_processed)

        with Pool(self.num_cpus) as pool:
            processing_tasks = pool.imap_unordered(
                self.process_task, not_processed
            )
            tasks_results = it.chain(processed, processing_tasks)

            for result in self._get_apply_bar(tasks_results, num_tasks):
                path, text, is_error = self._get_savinginfo(result)

                if not is_error:
                    texts.append((path, text))
                else:
                    errors.append((path, text))

        return self._to_df(texts, errors)

    def _apply_tasks(self, tasks):
        processed, not_processed = self._split_processed_tasks(tasks)
        num_tasks = len(tasks)
        tasks = (processed, not_processed)

        if self.chunksize is None:
            chunk_by_cpu = (len(not_processed)/self.num_cpus) / 100
            max_chunksize = self.max_docs_memory // self.num_cpus
            self.chunksize = int(max(1, min(chunk_by_cpu, max_chunksize)))

        if len(processed):
            logging.warning(
                f"Skipping {len(processed)} already"
                f" processed pages in directory '{self.tmp_dir}'"
            )

        if self.small:
            return self._apply_small(tasks, num_tasks)

        return self._apply_big(tasks, num_tasks)

    def apply(self):
        docs = self.get_docs(self.input_dir)
        tasks = self._gen_tasks(docs)

        return self._apply_tasks(tasks)
