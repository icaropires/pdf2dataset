import pytest
from pathlib import Path
import pandas as pd


SAMPLES_DIR = Path('tests/samples')
SAMPLE_IMAGE = SAMPLES_DIR / 'single_page1_1.jpeg'
PARQUET_ENGINE = 'pyarrow'


@pytest.fixture
def complete_df():

    def read_image(path, page):
        if page == -1:
            return None

        path = Path(path).with_suffix('')
        image_name = f'{path}_{page}.jpeg'
        image_path = Path(SAMPLES_DIR) / image_name

        with open(image_path, 'rb') as f:
            image_bin = f.read()

        return image_bin

    rows = [
        ['path', 'page', 'text', 'error_bool'],

        ['multi_page1.pdf', 1, 'First page', False],
        ['multi_page1.pdf', 2, 'Second page', False],
        ['multi_page1.pdf', 3, 'Third page', False],
        ['sub1/copy_multi_page1.pdf', 1, 'First page', False],
        ['sub1/copy_multi_page1.pdf', 2, 'Second page', False],
        ['sub1/copy_multi_page1.pdf', 3, 'Third page', False],
        ['single_page1.pdf', 1, 'My beautiful sample!', False],
        ['sub2/copy_single_page1.pdf', 1, 'My beautiful sample!', False],
        ['invalid1.pdf', -1, None, True]
    ]

    names = rows.pop(0)
    expected_dict = {n: r for n, r in zip(names, zip(*rows))}

    df = pd.DataFrame(expected_dict)
    df['image'] = df.apply(lambda row: read_image(row.path, row.page), axis=1)

    return df
