from pathlib import Path

from .extraction import Extraction
from .extraction_task import ExtractionTask


class ExtractionFromMemory(Extraction):

    def __init__(self, tasks, *args, **kwargs):
        self.tasks = tasks
        kwargs['check_inputdir'] = False

        super().__init__('', *args, **kwargs)

    def _gen_tasks(self, tasks):
        return self._gen_extrationtasks(tasks)  # Just to make more semantic

    def _gen_extrationtasks(self, tasks):
        ''' Generate ExtractionTask from simplified tasks.

        Assumes is not a big volume, otherwise should save documents to
        a directory and use 'Extraction'. So, not going with
        multiprocessing here.
        '''

        def uniform(task):
            page = None

            if len(task) == 2:
                doc, doc_bin = task
            elif len(task) == 3:
                doc, doc_bin, page = task
            else:
                raise RuntimeError(
                    'Wrong task format, it must be'
                    ' (document_name, document_bin)'
                    ' or (document_name, document_bin, page_number)'
                )

            if not str(doc).endswith('.pdf'):
                raise RuntimeError(
                    f"Document '{doc}' name must ends with '.pdf'"
                )

            range_pages = self._get_pages_range(doc, doc_bin=doc_bin)

            # -1 specifically because of the flag used by _get_pages_range
            if page in range_pages and not page == -1:
                range_pages = [page]

            elif page is not None:
                raise RuntimeError(
                    f"Page {page} doesn't exist in document {doc}!"
                )

            return Path(doc).resolve(), doc_bin, range_pages

        tasks = [uniform(t) for t in tasks]

        new_tasks = []
        for doc, doc_bin, range_pages in tasks:
            new_tasks += [
                ExtractionTask(doc, p, doc_bin, **self.task_params)
                for p in range_pages
            ]

        return new_tasks

    def apply(self):
        # Simplified notation tasks to ExtractionTasks
        tasks = self._gen_tasks(self.tasks)

        return self._process_tasks(tasks)
