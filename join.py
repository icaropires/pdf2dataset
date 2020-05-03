import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm


def read_texts(dir_path, *, workers=min(32, os.cpu_count() + 4)):

    dir_path = Path(dir_path)
    text_files = list(dir_path.rglob('*.txt'))

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


def getandsave_texts(dir_path, *, pickle_file='texts.pickle'):
    pickle_file = Path(pickle_file)

    if pickle_file.exists():
        print(f"'{pickle_file}' exists, using it...")
        texts = pickle.loads(pickle_file.read_bytes())
        return texts

    print('Searching for texts...')
    texts = read_texts(dir_path)

    print('Saving pickle with the result...')
    pickle_file.write_bytes(pickle.dumps(texts))
    print(f"New '{pickle_file}' saved!")

    return texts


if __name__ == '__main__':
    texts = getandsave_texts('txts')
    print('Texts available')
    # TODO: continue...
