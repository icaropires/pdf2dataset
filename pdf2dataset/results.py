from pathlib import Path
import pyarrow as pa

import pandas as pd
import dask.dataframe as dd


class Results:

    def __init__(self, input_dir, out_file, schema, *, max_size=None):
        self.input_dir = input_dir
        self.out_file = out_file
        self.schema = schema
        self.max_size = max_size

        self.results_dicts = []

    def __len__(self):
        return len(self.results_dicts)

    def _path_to_relative(self, file_path):
        if file_path is None:
            return None

        file_path = Path(file_path).resolve()
        input_dir = self.input_dir or Path().resolve()

        try:
            file_path = file_path.relative_to(input_dir)
        except ValueError:
            ...

        return str(file_path)

    def _results_as_df(self):
        if not self.results_dicts:
            table = pa.Table.from_arrays(
                [[]]*len(self.schema),
                schema=self.schema
            )

            return table.to_pandas()

        df = pd.DataFrame(self.results_dicts)
        df['path'] = df['path'].apply(self._path_to_relative)

        return df

    def write_and_clear(self):
        if self.results_dicts == []:
            raise RuntimeError('Trying to write empty results!')

        if self.out_file is None:
            raise ValueError("For writting 'out_file' must be provided")

        df = self._results_as_df()
        ddf = dd.from_pandas(df, npartitions=1)

        exists = self.out_file.exists()

        # Dask has some optimizations over pure pyarrow,
        #   like handling _metadata
        ddf.to_parquet(
            self.out_file, compression='gzip',
            ignore_divisions=True, write_index=False,
            schema=self.schema, append=exists, engine='pyarrow'
        )
        self.results_dicts = []

    def append(self, results_dicts):
        results_dicts = list(results_dicts)

        self.results_dicts.extend(results_dicts)

        if (self.max_size is not None and
                len(self.results_dicts) >= self.max_size):
            self.write_and_clear()

    def get(self):
        if self.max_size:
            raise RuntimeError(
                "Can't get() when using 'max_size' option,"
                " results would be incomplete"
            )

        return self._results_as_df()

    def is_tasks_processed(self, tasks):
        def gen_is_processed(df):
            num_checked = 0
            all_tasks = set(tuple(task) for task in df.itertuples(index=False))

            for task in tasks:
                if num_checked == len(tasks):
                    break  # All processed tasks were counted

                path = self._path_to_relative(task.path)

                is_processed = (path, task.page) in all_tasks
                num_checked += int(is_processed)

                yield is_processed

        try:
            df = pd.read_parquet(
                self.out_file,
                engine='pyarrow',
                columns=['path', 'page']
            )
        except (FileNotFoundError, ValueError):
            return (False for _ in tasks)

        return gen_is_processed(df)
