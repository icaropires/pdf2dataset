import io
import itertools as it
import logging
import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_EXCEPTION, as_completed
import queue
from threading import Thread
from multiprocessing import Pool, Process, Queue, Value
from pathlib import Path
from more_itertools import ichunked
from collections import deque
from functools import partial

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
        small=False, check_input=True, chunk_size=None,
        saving_interval=5000, max_files_memory=1000, files_pattern='*.pdf',

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
        self.chunk_size = chunk_size
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

        self.tasks_queue = Queue()
        self.loaded_tasks_queue = Queue(max_files_memory)

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

        self.files_list = self.search_files()
        return self.files_list

    def list_columns(self):
        aux_task = self.task_class(1, 1, **self.task_params)
        columns = aux_task.sel_features

        begin = list(aux_task.fixed_featues)
        columns = begin + sorted(c for c in columns if c not in begin)

        columns.append('error')  # Always last

        return columns

    def search_files(self):
        pdf_files = self.input_dir.rglob(self.files_pattern)

        # Here feedback is better than keeping use of the generator
        return list(tqdm(pdf_files, desc='Looking for files', unit='files'))

    def _load_tasks(self):
        def load_and_put(task):
            task = task.copy()
            task.load_bin()

            self.loaded_tasks_queue.put(task)

        futures = []
        with ThreadPoolExecutor() as executor:
            for task in iter(self.tasks_queue.get, None):
                future = executor.submit(load_and_put, task)
                futures.append(future)

            for future in futures:
                future.result()

        self.loaded_tasks_queue.put(None)

    def _gen_tasks_proc(self, total_tasks):
        load_thread = Thread(target=self._load_tasks)
        load_thread.start()

        gen_tasks = partial(
            self.gen_tasks,
            task_class=self.task_class, task_params=self.task_params
        )

        with Pool(self.num_cpus) as pool:
            mapping = pool.imap_unordered(gen_tasks, self.files)
            tasks = it.chain.from_iterable(mapping)

            for task in tasks:
                self.tasks_queue.put(task)
                total_tasks.value += 1

        self.tasks_queue.put(None)
        load_thread.join()

    @classmethod
    def gen_tasks(cls, path, task_class, task_params):
        '''
        Returns tasks to be processed.
        Can't be instance class, because of the pool
        For faulty files, only the page -1 will be available
        '''
        def gen_task(page):
            file_bin = path.read_bytes()
            return task_class(path, page, **task_params)

        return [gen_task(page) for page in cls.get_pages_range(path)]

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
            iterable, total=num_tasks+num_skipped, initial=num_skipped,
            desc='Processing pages', unit='pages', dynamic_ncols=True
        )

    @staticmethod
    @ray.remote
    def _process_chunk_ray(chunk):
        return [t.process() for t in chunk]

#    def _ray_producer(self, loaded_tasks_queue, refs_queue):
#        loaded_tasks = iter(loaded_tasks_queue.get, None)
#        chunks = ichunked(loaded_tasks, self.chunk_size)
#
#        import timeit
#        c = 0
#        n = 0
#        for chunk in chunks:
#            start = timeit.default_timer()
#            future = self._process_chunk_ray.remote(list(chunk))
#            c += timeit.default_timer() - start
#            n += 1
#            
#            refs_queue.put(future)
#
#            if n % 100 == 0:
#                print('@@@@@@@@@@@@@@@@@')
#                print(c/n)
#                print('@@@@@@@@@@@@@@@@@')
#        refs_queue.put(None)

    def _ray_producer(self, loaded_tasks_queue, refs_queue):
        loaded_tasks = iter(loaded_tasks_queue.get, None)
        submit_fn = self._process_chunk_ray.remote

        has_finished = False

        with ThreadPoolExecutor() as executor:
            while not has_finished:
                futures = []

                for _ in range(refs_queue.maxsize):
                    chunk = list(it.islice(loaded_tasks, self.chunk_size))

                    if not chunk:
                        has_finished = True
                        break

                    future = executor.submit(submit_fn, chunk)
                    futures.append(future)

                for future in as_completed(futures):
                    refs_queue.put(future.result())

            refs_queue.put(None)

    def _ray_consumer(self, results_queue, refs_queue):
        for future in iter(refs_queue.get, None):
            (finished, *_), rest = ray.wait([future], num_returns=1)
            result = ray.get(finished)

            results_queue.put(result)

        results_queue.put(None)

    def _ray_process_proc(self, loaded_tasks_queue, results_queue):
        ray.init(**self.ray_params)

        max_futures = int(ray.available_resources()['CPU']) * 3  # TODO: create variable
        refs_queue = queue.Queue(max_futures)

        # Not using 'with' because was blocking exceptions to be raised
        executor = ThreadPoolExecutor(2)

        producer = executor.submit(self._ray_producer, loaded_tasks_queue, refs_queue)
        consumer = executor.submit(self._ray_consumer,  results_queue, refs_queue)

        for f in as_completed([producer, consumer]):
            f.result()

        executor.shutdown()

    def _update_bar_total(self, progress_bar, total_tasks):
        while True:
            progress_bar.total = total_tasks.value
            progress_bar.refresh()

            sleep(0.2)

    def _apply_to_big(self, tasks):
        'Apply the extraction to a big volume of data'

        print('\n=== SUMMARY ===',
              f'PDFs directory: {self.input_dir}',
              f'Results file: {self.out_file}',
              f'Using {self.num_cpus} CPU(s)',
              f'Chunksize: {self.chunk_size}',
              sep='\n', end='\n\n')

        results_queue = Queue()
        total_tasks = Value('i', 0)

        with self._get_processing_bar(0) as progress_bar:
            Thread(target=self._update_bar_total, args=(progress_bar, total_tasks), daemon=True).start()

            gen_tasks_proc = Process(target=self._gen_tasks_proc, args=(total_tasks,))
            ray_proc = Process(target=self._ray_process_proc, args=(self.loaded_tasks_queue, results_queue))

            gen_tasks_proc.start()
            ray_proc.start()

            try:
                for result_chunk in iter(results_queue.get, None):
                    if isinstance(result_chunk, Exception):
                        raise result_chunk

                    self.results.append(result_chunk)
                    progress_bar.update(len(result_chunk))

                if self.results:
                    self.results.write_and_clear()

                gen_tasks_proc.join()
                ray_proc.join()

            except KeyboardInterrupt:
                gen_tasks_proc.terminate()
                ray_proc.terminate()

            finally:
                gen_tasks_proc.close()
                ray_proc.close()

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
        self.chunk_size = 40
        # if self.chunk_size is None:
        #     chunk_by_cpu = (len(tasks)/self.num_cpus) / 100
        #     max_chunksize = self.max_files_memory // self.num_cpus
        #     self.chunk_size = int(max(1, min(chunk_by_cpu, max_chunksize)))

        # if self.small:
        #     return self._apply_to_small(tasks)

        self._apply_to_big(tasks)

        return self.out_file

    def apply(self):
        # num_total_tasks = len(tasks)
        # tasks = self.filter_processed_tasks(tasks)
        # self.num_skipped = num_total_tasks - len(tasks)

        # if self.num_skipped:
        #     logging.warning(
        #         "'%s' have already %d processed pages, skipping these...",
        #         self.out_file, self.num_skipped
        #     )

        return self._process_tasks(None)
