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
import dask.dataframe as dd
import ray
from tqdm import tqdm
from pathlib import Path
import pdftotext

from .extraction_task import ExtractionTask
from .constants import POSSIBLE_FEATURES  # TODO: calculate


# TODO: Some day I will reduce this file size!!

# TODO: Add typing
# TODO: Set up a logger to the class
# TODO: Substitute most (all?) prints for logs
# TODO: Create a 'task result' namedtuple


class TextExtraction:
    _path_pat = r'(?P<path>.+)_(?P<feature>(\w|-)+)_(?P<page>-?\d+)\.txt'

    def __init__(
        self, input_dir, results_file='', *,
        tmp_dir='', lang='por', ocr=False, small=False, features='text',
        img_size=None, chunksize=None, chunk_df_size=10000,
        check_inputdir=True, max_docs_memory=3000, **ray_params
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
        self.features = self._parse_featues(features)
        self.img_size = img_size.lower() if img_size is not None else None
        self.max_docs_memory = max_docs_memory
        self.chunk_df_size = chunk_df_size

        self._df_lock = threading.Lock()
        self._validate_features_param()

    @staticmethod
    def _parse_featues(features):
        if features == '':
            return []

        if features != 'all':
            return features.split(',')

        not_extracted_features = [
            f for f in POSSIBLE_FEATURES if f
            not in ['path', 'doc', 'page', 'error']
        ]

        return not_extracted_features

    def _validate_features_param(self):
        for feature in self.features:
            assert feature in POSSIBLE_FEATURES, f'Invalid feature: {feature}'

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

    def _get_feature_path(self, doc, page, feature):
        output_path = self._get_output_path(Path(doc))
        basename = f'{output_path.stem}_{feature}_{page}.txt'

        return output_path.with_name(basename)

    @staticmethod
    def _preprocess_path(df):
        parsed = df['path'].str.extract(TextExtraction._path_pat)
        df['path'] = parsed['path'] + '.pdf'

        return df

    def _doc_to_path(self, df):
        any_feature = 'doc'

        def gen_path(row):
            path = self._get_feature_path(row.doc, row.page, any_feature)
            return str(path)

        df['path'] = df.apply(gen_path, axis=1)
        df.pop('doc')

        return df

    def _save_tmp_files(self, result):
        for feature in result:
            if feature in ['doc', 'page']:
                continue

            # TODO: save any format, not just text
            doc, page = result['doc'], result['page']
            path = self._get_feature_path(doc, page, feature)
            content = result[feature] or ''

            Path(path).write_text(content)

    def _to_df(self, results):
        df = pd.DataFrame(results)

        df = self._doc_to_path(df)
        df = self._preprocess_path(df)

        # Keep path and page as first columns
        begin = ['path', 'page']
        columns = begin + [c for c in df.columns.to_list() if c not in begin]

        return df[columns]

    def _append_to_df(self, results):
        df = self._to_df(results)
        df = dd.from_pandas(df, npartitions=1)

        with self._df_lock:
            df.to_parquet(self.results_file, compression='gzip',
                          append=self.results_file.exists(), engine='pyarrow')

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
                    ExtractionTask(
                        doc, p, lang=self.lang, ocr=self.ocr,
                        features=self.features, img_size=self.img_size
                    )
                    for p in range_pages
                ]
                tasks += new_tasks
                pbar.update(len(new_tasks))

        return tasks

    def _get_features_list(self, sample_task):
        sample_task = ([], [sample_task])
        sample_result = self._apply_to_small(sample_task, 1, progress=False)

        return sample_result.columns

    def _load_procesed_tasks(self, processed):
        if not processed:
            return []

        features = self._get_features_list(processed[0])

        def load(task):
            paths = {f: self._get_feature_path(task.doc, task.page, f)
                     for f in features if f not in ['path', 'page']}

            result = {f: p.read_text() or None for f, p in paths.items()}
            result['doc'] = task.doc
            result['page'] = task.page

            return result

        return [load(task) for task in processed]

    def _split_processed_tasks(self, tasks):
        features = self._get_features_list(tasks[0])

        processed, not_processed = [], []
        for task in tasks:
            paths = (self._get_feature_path(task.doc, task.page, f)
                     for f in features if f not in ['path', 'page'])

            if all(p.exists() for p in paths):
                processed.append(task)
            else:
                not_processed.append(task)

        return processed, not_processed

    @staticmethod
    def _get_pbar(tasks_results, num_tasks):
        return tqdm(
            tasks_results, total=num_tasks,
            desc='Processing pages', unit='pages', dynamic_ncols=True
        )

    @staticmethod
    def _load_task_bin(task):
        task = task.copy()  # Copy to have control over memory references
        task.load_bin()

        return task

    @ray.remote
    def process_chunk_ray(chunk):
        return [t.process() for t in chunk]

    def _ray_process(self, tasks):
        tasks = iter(tasks)
        futures = []

        chunks = ichunked(tasks, int(self.chunksize))

        for chunk in chunks:
            chunk = [self._load_task_bin(task) for task in chunk]
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
            except StopIteration:
                ...
            else:
                chunk = [self._load_task_bin(t) for t in chunk]
                rest.append(self.process_chunk_ray.remote(chunk))

            futures = rest

    def _apply_to_big(self, tasks, num_tasks, progress=True):
        'Apply the extraction to a big volume of data'

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.results_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {self.chunksize}',
              f'Temporary directory: {self.tmp_dir or "not used"}',
              sep='\n', end='\n\n')

        # TODO: Add queue
        with ThreadPoolExecutor(max_workers=4) as e:
            processed, not_processed = tasks

            not_processed = self._ray_process(not_processed)
            processing_tasks = it.chain(processed, not_processed)

            if progress:
                processing_tasks = self._get_pbar(processing_tasks, num_tasks)

            thread_fs, results = [], []
            for result in processing_tasks:
                if isinstance(result, Exception):
                    raise result

                results.append(result)

                if self.tmp_dir:
                    f = e.submit(self._save_tmp_files, result)
                    thread_fs.append(f)

                if len(results) >= self.chunk_df_size:
                    # Persist to disk, aiming large amount of data
                    f = e.submit(self._append_to_df, results)
                    thread_fs.append(f)

                    results = []

            if results:
                f = e.submit(self._append_to_df, results)
                thread_fs.append(f)

            for f in thread_fs:  # Avoid fail silently
                f.result()

    @staticmethod
    def _process_task(task):
        return task.process()

    def _apply_to_small(self, tasks, num_tasks, progress=True):
        ''''Apply the extraction to a small volume of data
        More direct approach than 'big', but with these differences:
            - Not saving progress
            - Distributed processing not supported
            - Don't write dataframe to disk
            - Returns the resultant dataframe
        '''

        processed, not_processed = tasks
        not_processed = (self._load_task_bin(t) for t in not_processed)

        results = []
        with Pool(self.num_cpus) as pool:
            processing_tasks = pool.imap_unordered(self._process_task,
                                                   not_processed)
            processing_tasks = it.chain(processed, processing_tasks)

            if progress:
                processing_tasks = self._get_pbar(processing_tasks, num_tasks)

            results = list(processing_tasks)

        return self._to_df(results)

    def _process_tasks(self, tasks):
        processed, not_processed = self._split_processed_tasks(tasks)
        processed = self._load_procesed_tasks(processed)

        num_tasks = len(tasks)
        tasks = (processed, not_processed)

        if len(processed):
            logging.warning(
                f"Skipping {len(processed)} already"
                f" processed pages in directory '{self.tmp_dir}'"
            )

        if self.chunksize is None:
            chunk_by_cpu = (len(not_processed)/self.num_cpus) / 100
            max_chunksize = self.max_docs_memory // self.num_cpus
            self.chunksize = int(max(1, min(chunk_by_cpu, max_chunksize)))

        if self.small:
            return self._apply_to_small(tasks, num_tasks)

        result = None
        try:
            ray.init(**self.ray_params)
            result = self._apply_to_big(tasks, num_tasks)
        finally:
            ray.shutdown()

        return result

    def apply(self):
        docs = self.get_docs(self.input_dir)
        tasks = self._gen_tasks(docs)

        return self._process_tasks(tasks)
