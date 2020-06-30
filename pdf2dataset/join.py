import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm


def read_texts(dir_path, *,
               workers=min(32, os.cpu_count() + 4),
               pattern='*.txt'):

    dir_path = Path(dir_path)
    text_files = list(dir_path.rglob(pattern))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=len(text_files),
                  unit='docs', desc='Joining texts...') as pbar:

            def submit(text_file):
                future = executor.submit(text_file.read_text)
                future.add_done_callback(lambda _: pbar.update())

                return future

            texts = {path.resolve(): f.result() for path, f in
                     [(t, submit(t)) for t in text_files]}

    return texts


def join(dir_path, *,
         results_file='texts.pickle',
         error_file='errors.log',
         workers=min(32, os.cpu_count() + 4),
         error_suffix='_error.log'):

    results_file = Path(results_file)
    error_file = Path(error_file)

    if results_file.exists():
        print(f"'{results_file}' exists, using it...")
        texts = pickle.loads(results_file.read_bytes())
        return texts

    print('Searching for texts...')
    texts = read_texts(dir_path, workers=workers)

    error_pat = f'*{error_suffix}'
    errors = read_texts(dir_path, pattern=error_pat, workers=workers)
    errors = '\n\n'.join(f'{path}: {error}' for path, error in errors.items())

    print('Saving files...')
    results_file.write_bytes(pickle.dumps(texts))
    error_file.write_text(errors)
    print(f"New '{results_file}' and '{error_file}' saved!")

    return texts


if __name__ == '__main__':
    texts = join('/tmp/text')
    print('Texts available')
    # TODO: continue...
