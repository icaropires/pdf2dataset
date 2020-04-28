import os
import pickle
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

FOLDER_PATH = Path('/tmp/pdfs/')


def get_text(path):
    with path.open() as f:
        text = f.read()

    return path, text


def search_texts():
    text_files = list(FOLDER_PATH.glob('**/*.txt'))

    texts = {}
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        with tqdm(total=len(text_files), desc='Reading files...') as pbar:
            futures = []
            for text_file in text_files:
                future = executor.submit(get_text, text_file)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)

            for future in futures:
                path, text = future.result()
                texts[path.resolve()] = text

    return texts


def get_texts():
    pickle_file = Path('texts.pickle')

    if not pickle_file.exists():
        print('Searching for texts...')
        texts = search_texts()

        print('Saving pickle with the result...')
        with pickle_file.open('wb') as f:
            pickle.dump(texts, f)

        print(f'New {pickle_file} saved!')
    else:
        print(f'{pickle_file} exists, using it...')

        with pickle_file.open('rb') as f:
            texts = pickle.load(f)

        return texts


texts = get_texts()
print('Texts available')
# TODO: continue...
