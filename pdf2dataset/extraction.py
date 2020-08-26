import io
import itertools as it
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Process, Queue
from pathlib import Path
from more_itertools import ichunked

import ray
import pdftotext
from tqdm import tqdm
from pytesseract import get_tesseract_version

from .pdf_extract_task import PdfExtractTask
from .results import Results


# TODO: Eventually, I'll reduce this file size!!

# TODO: Add typing?
# TODO: Set up a logger to the class
# TODO: Substitute most (all?) prints for logs


def get_pages_range(path, doc_bin=None):
    # Using pdftotext to get num_pages because it's the best way I know
    # pdftotext extracts lazy, so this won't process the text

    try:
        if not doc_bin:
            with path.open('rb') as f:
                num_pages = len(pdftotext.PDF(f))
        else:
            with io.BytesIO(doc_bin) as f:
                num_pages = len(pdftotext.PDF(f))

        pages = range(1, num_pages+1)
    except pdftotext.Error:
        pages = [-1]

    return pages


class Extraction:

    def __init__(
        self, input_dir, out_file='', *,
        small=False, check_inputdir=True, chunksize=None, chunk_df_size=10000,
        max_files_memory=3000, task_class=PdfExtractTask,
        files_pattern='*.pdf', files_list=None,

        # TODO: task params
        ocr=False, ocr_image_size=None, ocr_lang='por', features='all',
        image_format='jpeg', image_size=None,

        **ray_params
    ):
        self.input_dir = Path(input_dir).resolve()
        self.out_file = Path(out_file).resolve()

        if (check_inputdir and
                (not (self.input_dir.exists() and self.input_dir.is_dir()))):
            raise ValueError(f"Invalid input_dir: '{self.input_dir}',"
                             " it must exists and be a directory")

        if not small:
            if not out_file:
                raise ValueError("If not using 'small' arg,"
                                 " 'out_file' is mandatory")

            if self.out_file.exists():
                logging.warning(f'{out_file} already exists!'
                                ' Results will be appended to it!')

        if ocr:
            # Will raise exception if tesseract was not found
            get_tesseract_version()

        self.num_cpus = ray_params.get('num_cpus') or os.cpu_count()
        self.ray_params = ray_params
        self.chunksize = chunksize
        self.small = small
        self.max_files_memory = max_files_memory
        self.files_pattern = files_pattern
        self.files_list = files_list

        self.task_class = task_class
        self.task_params = {
            'sel_features': features,
            'ocr': ocr,
            'ocr_lang': ocr_lang,
            'ocr_image_size': ocr_image_size,
            'image_format': image_format,
            'image_size': image_size,
        }

        columns = self.list_columns()
        schema = self.task_class.get_schema(columns)

        max_results_size = chunk_df_size if not small else None
        self.results = Results(
            self.input_dir, self.out_file, schema, max_size=max_results_size
        )

        self.results_queue = Queue(max_files_memory)

    def list_columns(self):
        aux_task = self.task_class(1, 1, **self.task_params)

        columns = aux_task.sel_features

        begin = list(aux_task.fixed_featues)
        columns = begin + sorted(c for c in columns if c not in begin)

        columns.append('error')  # Always last

        return columns

    def list_files(self, input_dir):
        pdf_files = input_dir.rglob(self.files_pattern)

        # Here feedback is better than keeping use of the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='files'))

    def _gen_tasks(self, files):
        '''
        Returns tasks to be processed.
        For faulty documents, only the page -1 will be available
        '''
        # 10 because this is a fast operation
        chunksize = int(max(1, (len(files)/self.num_cpus)//10))

        tasks = []
        with Pool(self.num_cpus) as pool, \
                tqdm(desc='Counting pages', unit='pages') as pbar:

            results = pool.imap(
                get_pages_range, files, chunksize=chunksize
            )

            for path, range_pages in zip(files, results):
                new_tasks = [
                    self.task_class(path, p, **self.task_params)
                    for p in range_pages
                ]
                tasks += new_tasks
                pbar.update(len(range_pages))

        return tasks

    @staticmethod
    def _get_processing_bar(num_tasks, iterable=None):
        return tqdm(
            iterable, total=num_tasks,
            desc='Processing pages', unit='pages', dynamic_ncols=True
        )

    @staticmethod
    def _load_task_bin(task):
        task = task.copy()  # Copy to have control over memory references
        task.load_bin()

        return task

    @staticmethod
    @ray.remote
    def _process_chunk_ray(chunk):
        return [t.process() for t in chunk]

    @staticmethod
    def _submit_chunk_ray(chunk):
        with ThreadPoolExecutor() as executor:
            chunk = list(executor.map(Extraction._load_task_bin, chunk))

        return Extraction._process_chunk_ray.remote(chunk)

    def _ray_process_aux(self, tasks, results_queue):
        chunks = ichunked(tasks, int(self.chunksize))
        num_initial = int(ray.available_resources()['CPU'])

        futures = [self._submit_chunk_ray(c)
                   for c in it.islice(chunks, num_initial)]

        with self._get_processing_bar(len(tasks)) as progress_bar:
            while futures:
                (finished, *_), rest = ray.wait(futures, num_returns=1)

                result = ray.get(finished)
                results_queue.put(result)

                progress_bar.update(len(result))

                try:
                    chunk = next(chunks)
                except StopIteration:
                    ...
                else:
                    rest.append(self._submit_chunk_ray(chunk))

                futures = rest

        results_queue.put(None)

    def _ray_process(self, *args, **kwargs):
        ray.init(**self.ray_params)

        try:
            self._ray_process_aux(*args, **kwargs)
        finally:
            ray.shutdown()

    def _apply_to_big(self, tasks, num_tasks):
        'Apply the extraction to a big volume of data'

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.out_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {self.chunksize}',
              sep='\n', end='\n\n')

        processed, not_processed = tasks

        # processing_tasks = it.chain(processed, not_processed)

        Process(
            target=self._ray_process,
            args=(not_processed, self.results_queue)
        ).start()

        while True:
            result_chunk = self.results_queue.get()

            if isinstance(result_chunk, Exception):
                raise result_chunk

            if result_chunk is None:
                break

            self.results.append(result_chunk)

        if len(self.results):
            self.results.write_and_clear()

    @staticmethod
    def _process_task(task):
        return task.process()

    def _apply_to_small(self, tasks, num_tasks):
        ''''Apply the extraction to a small volume of data
        More direct approach than 'big', but with these differences:
            - Not saving progress
            - Distributed processing not supported
            - Don't write dataframe to disk
            - Returns the resultant dataframe
        '''

        processed, not_processed = tasks
        not_processed = (self._load_task_bin(t) for t in not_processed)

        with Pool(self.num_cpus) as pool:
            processing_tasks = pool.imap_unordered(
                self._process_task, not_processed
            )

            processing_tasks = it.chain(processed, processing_tasks)

            processing_tasks = self._get_processing_bar(
                num_tasks, processing_tasks
            )

            self.results.append(list(processing_tasks))

        return self.results.get()

    def _process_tasks(self, tasks):
        processed, not_processed = self.results.get_processed_tasks(tasks)
        # processed = self._load_procesed_tasks(processed)

        num_tasks = len(tasks)
        tasks = (processed, not_processed)

        if processed:
            logging.warning(
                # TODO
                f"Skipping {len(processed)} already"  # tmp_dir
                # f" processed pages in directory '{self.tmp_dir}'"
            )

        if self.chunksize is None:
            chunk_by_cpu = (len(not_processed)/self.num_cpus) / 100
            max_chunksize = self.max_files_memory // self.num_cpus
            self.chunksize = int(max(1, min(chunk_by_cpu, max_chunksize)))

        if self.small:
            return self._apply_to_small(tasks, num_tasks)

        self._apply_to_big(tasks, num_tasks)

        return self.out_file

    def apply(self):
        if self.files_list is None:
            files = self.list_files(self.input_dir)
        else:
            files = self.files_list

        tasks = self._gen_tasks(files)

        return self._process_tasks(tasks)
