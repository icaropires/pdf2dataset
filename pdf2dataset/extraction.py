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


class Extraction:

    def __init__(
        self, input_dir=None, out_file=None, *,
        files_list=None, task_class=PdfExtractTask,

        # Config params
        small=False, check_input=True, chunksize=None,
        saving_interval=5000, max_files_memory=3000, files_pattern='*.pdf',

        # Task_params
        ocr=False, ocr_image_size=None, ocr_lang='por', features='all',
        image_format='jpeg', image_size=None,

        **ray_params
    ):
        self.input_dir = Path(input_dir).resolve() if input_dir else None
        self.files_list = [Path(f) for f in files_list] if files_list else None

        self.out_file = Path(out_file).resolve() if out_file else None

        if check_input:
            self._check_input()

        if not small:
            self._check_outfile()

        if ocr:
            # Will raise exception if tesseract was not found
            get_tesseract_version()

        self.num_cpus = ray_params.get('num_cpus') or os.cpu_count()
        self.ray_params = ray_params
        self.chunksize = chunksize
        self.small = small
        self.max_files_memory = max_files_memory
        self.files_pattern = files_pattern

        self.num_skipped = None

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

        max_results_size = saving_interval if not small else None
        self.results = Results(
            self.input_dir, self.out_file, schema, max_size=max_results_size
        )

        self.results_queue = Queue(max_files_memory)

    def _check_input(self):
        if not any([self.input_dir, self.files_list]):
            raise RuntimeError(
                "Any of 'input_dir' or 'self.files_list' must be provided"
            )

        if not self.files_list:
            self._check_inputdir()

    def _check_inputdir(self):
        if (not self.input_dir
                or not (self.input_dir.exists() and self.input_dir.is_dir())):

            raise ValueError(f"Invalid input_dir: '{self.input_dir}',"
                             " it must exists and be a directory")

    def _check_outfile(self):
        if not self.out_file:
            raise RuntimeError("If not using 'small' arg,"
                               " 'out_file' is mandatory")

    @property
    def files(self):
        if self.files_list:
            return self.files_list

        self.files_list = self.list_files()
        return self.files_list

    def list_columns(self):
        aux_task = self.task_class(1, 1, **self.task_params)

        columns = aux_task.sel_features

        begin = list(aux_task.fixed_featues)
        columns = begin + sorted(c for c in columns if c not in begin)

        columns.append('error')  # Always last

        return columns

    def list_files(self):
        pdf_files = self.input_dir.rglob(self.files_pattern)

        # Here feedback is better than keeping use of the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='files'))

    def gen_tasks(self):
        '''
        Returns tasks to be processed.
        For faulty files, only the page -1 will be available
        '''
        # 10 because this is a fast operation
        chunksize = int(max(1, (len(self.files)/self.num_cpus)//10))

        tasks = []
        with Pool(self.num_cpus) as pool, \
                tqdm(desc='Counting pages', unit='pages') as pbar:

            results = pool.imap(
                self.get_pages_range, self.files, chunksize=chunksize
            )

            for path, range_pages in zip(self.files, results):
                new_tasks = [
                    self.task_class(path, p, **self.task_params)
                    for p in range_pages
                ]
                tasks += new_tasks
                pbar.update(len(range_pages))

        return tasks

    @staticmethod
    def get_pages_range(file_path, file_bin=None):
        # Using pdftotext to get num_pages because it's the best way I know
        # pdftotext extracts lazy, so this won't process the text

        try:
            if not file_bin:
                with file_path.open('rb') as f:
                    num_pages = len(pdftotext.PDF(f))
            else:
                with io.BytesIO(file_bin) as f:
                    num_pages = len(pdftotext.PDF(f))

            pages = range(1, num_pages+1)
        except pdftotext.Error:
            pages = [-1]

        return pages

    def _get_processing_bar(self, num_tasks, iterable=None):
        num_skipped = self.num_skipped or 0

        return tqdm(
            iterable, total=num_tasks, initial=num_skipped,
            desc='Processing pages', unit='pages', dynamic_ncols=True
        )

    @staticmethod
    def copy_and_load_task(task):
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
            chunk = list(executor.map(Extraction.copy_and_load_task, chunk))

        return Extraction._process_chunk_ray.remote(chunk)

    def _ray_process_aux(self, tasks, results_queue):
        chunks = ichunked(tasks, int(self.chunksize))
        num_initial = int(ray.available_resources()['CPU'])

        with self._get_processing_bar(len(tasks)) as progress_bar:
            futures = [self._submit_chunk_ray(c)
                       for c in it.islice(chunks, num_initial)]

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

    def _apply_to_big(self, tasks):
        'Apply the extraction to a big volume of data'

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.out_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {self.chunksize}',
              sep='\n', end='\n\n')

        Process(
            target=self._ray_process,
            args=(tasks, self.results_queue)
        ).start()

        while True:
            result_chunk = self.results_queue.get()

            if isinstance(result_chunk, Exception):
                raise result_chunk

            if result_chunk is None:
                break

            self.results.append(result_chunk)

        if self.results:
            self.results.write_and_clear()

    @staticmethod
    def _process_task(task):
        return task.process()

    def _apply_to_small(self, tasks):
        ''''Apply the extraction to a small volume of data
        More direct approach than 'big', but with these differences:
            - Not saving progress
            - Distributed processing not supported
            - Don't write dataframe to disk
            - Returns the resultant dataframe
        '''

        num_tasks = len(tasks)
        tasks = (self.copy_and_load_task(t) for t in tasks)

        with Pool(self.num_cpus) as pool:
            processing_tasks = pool.imap_unordered(self._process_task, tasks)
            results = self._get_processing_bar(num_tasks, processing_tasks)

            self.results.append(results)

        return self.results.get()

    def filter_processed_tasks(self, tasks):
        is_processed = self.results.is_tasks_processed(tasks)
        tasks = [t for t, is_ in zip(tasks, is_processed) if not is_]

        return tasks

    def _process_tasks(self, tasks):
        if self.chunksize is None:
            chunk_by_cpu = (len(tasks)/self.num_cpus) / 100
            max_chunksize = self.max_files_memory // self.num_cpus
            self.chunksize = int(max(1, min(chunk_by_cpu, max_chunksize)))

        if self.small:
            return self._apply_to_small(tasks)

        self._apply_to_big(tasks)

        return self.out_file

    def apply(self):
        tasks = self.gen_tasks()

        num_total_tasks = len(tasks)
        tasks = self.filter_processed_tasks(tasks)
        self.num_skipped = num_total_tasks - len(tasks)

        if self.num_skipped:
            logging.warning(
                "'%s' have already %d processed pages, skipping these...",
                self.out_file, self.num_skipped
            )

        return self._process_tasks(tasks)
