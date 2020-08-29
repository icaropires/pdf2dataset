from pathlib import Path

from .extraction import Extraction


class ExtractionFromMemory(Extraction):

    def __init__(self, tasks, *args, **kwargs):
        self.tasks = tasks

        kwargs['check_input'] = False

        super().__init__(None, *args, **kwargs)

    def gen_tasks(self):
        # Just to make more semantic
        return self._gen_extrationtasks(self.tasks)

    def _gen_extrationtasks(self, tasks):
        ''' Generate extraction task from simplified tasks form.

        Assumes is not a big volume, otherwise should save files to
        a directory and use 'Extraction'. So, not going with
        multiprocessing here.
        '''

        def uniform(task):
            page = None

            if len(task) == 2:
                file_, file_bin = task
            elif len(task) == 3:
                file_, file_bin, page = task
            else:
                raise RuntimeError(
                    'Wrong task format, it must be'
                    ' (file_name, file_bin)'
                    ' or (file_name, file_bin, page_number)'
                )

            if not str(file_).endswith('.pdf'):
                raise RuntimeError(
                    f"Document '{file_}' name must ends with '.pdf'"
                )

            range_pages = self.get_pages_range(file_, file_bin=file_bin)

            # -1 specifically because of the flag used by _get_pages_range
            if page in range_pages and not page == -1:
                range_pages = [page]

            elif page is not None:
                raise RuntimeError(
                    f"Page {page} doesn't exist in file {file_}!"
                )

            return Path(file_).resolve(), file_bin, range_pages

        tasks = [uniform(t) for t in tasks]

        new_tasks = []
        for file_, file_bin, range_pages in tasks:
            new_tasks += [
                self.task_class(file_, p, file_bin, **self.task_params)
                for p in range_pages
            ]

        return new_tasks

    def apply(self):
        # Coverts simplified notation tasks to extraction tasks
        tasks = self.gen_tasks()

        return self._process_tasks(tasks)
