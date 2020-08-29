from io import BytesIO
from pathlib import Path
from hashlib import md5

import pytest
import pandas as pd
import numpy as np
from PIL import Image

from pdf2dataset import (
    ExtractionFromMemory,
    PdfExtractTask,
    extract,
    extract_text,
    image_to_bytes,
    image_from_bytes,
)

from .testing_dataframe import check_and_compare


SAMPLES_DIR = Path('tests/samples')
TEST_IMAGE = SAMPLES_DIR / 'single_page1_1.jpeg'
PARQUET_ENGINE = 'pyarrow'


@pytest.fixture
def expected_all():

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


@pytest.fixture
def image():
    return Image.open(TEST_IMAGE)


@pytest.fixture
def image_bytes():
    with open(TEST_IMAGE, 'rb') as f:
        bytes_ = f.read()

    return bytes_


class TestExtractionCore:

    @pytest.mark.parametrize('is_ocr', (
        True,
        False,
    ))
    def test_extraction_big(self, tmp_path, is_ocr, expected_all):
        result_path = tmp_path / 'result.parquet.gzip'

        extract(SAMPLES_DIR, result_path,
                ocr_lang='eng', ocr=is_ocr, features='all')

        df = pd.read_parquet(result_path, engine=PARQUET_ENGINE)

        if is_ocr:
            df['text'] = df['text'].str.strip()

        check_and_compare(df, expected_all, is_ocr=is_ocr)

    def test_append_result(self, tmp_path, expected_all):
        result_path = tmp_path / 'result.parquet.gzip'

        extract(SAMPLES_DIR, result_path, saving_interval=1, features='all')

        # Small 'chunk_df_size' to append to result multiple times
        df = pd.read_parquet(result_path, engine=PARQUET_ENGINE)

        check_and_compare(df, expected_all)

    def test_passing_paths_list(self, tmp_path, expected_all):
        result_path = tmp_path / 'result.parquet.gzip'
        files_list = Path(SAMPLES_DIR).rglob('*.pdf')

        # Test the support for paths as strings
        files_list = [str(f) for f in files_list]

        df = extract(files_list, result_path, small=True)

        # Paths will be relative to pwd, so adapting expected_all
        expected_all['path'] = expected_all['path'].apply(
            lambda p: str(SAMPLES_DIR / p)
        )
        check_and_compare(df, expected_all)

    def test_filter_processed(self, tmp_path):
        with open(SAMPLES_DIR / 'single_page1.pdf', 'rb') as f:
            single = f.read()

        with open(SAMPLES_DIR / 'multi_page1.pdf', 'rb') as f:
            multi = f.read()

        with open(SAMPLES_DIR / 'invalid1.pdf', 'rb') as f:
            invalid = f.read()

        total_tasks = [
            ('single1.pdf', single),
            ('multi1.pdf', multi, 2),
            ('hey/multi2.pdf', multi, 1),
            ('multi1.pdf', multi, 1),
            ('my_dir/single3.pdf', single),
            ('/opt/invalid.pdf', invalid),
            ('multi1.pdf', multi, 3),
            ('single2.pdf', single),
            ('/tmp/single3.pdf', single),
            ('/tmp/invalid.pdf', invalid),
        ]

        result_path = tmp_path / 'result.parquet.gzip'

        for counter, task in enumerate(total_tasks):

            extraction = ExtractionFromMemory(
                total_tasks,
                out_file=result_path,
                features='text'
            )

            tasks = extraction.gen_tasks()
            tasks = extraction.filter_processed_tasks(tasks)

            assert len(tasks) == len(total_tasks) - counter

            extract([task], result_path, features='text')


class TestExtractionSmall:
    @pytest.mark.parametrize('is_ocr', (
        True,
        False,
    ))
    def test_extraction_small(self, is_ocr, expected_all):
        df = extract(SAMPLES_DIR, small=True, ocr_lang='eng', ocr=is_ocr)

        if is_ocr:
            df['text'] = df['text'].str.strip()

        check_and_compare(df, expected_all, is_ocr=is_ocr)

    def test_return_list(self):
        def sort(doc):
            try:
                first_page_idx, text_idx = 0, 1

                return len(doc[first_page_idx][text_idx])
            except TypeError:
                return -1  # For None values

        def hash_images(doc):
            for page_idx, page in enumerate(doc):
                image, text = page
                image = md5(image).hexdigest() if image is not None else None

                doc[page_idx] = [image, text]

            return doc

        list_ = extract(SAMPLES_DIR, return_list=True)
        list_ = [hash_images(doc) for doc in list_]

        # Expected structure:
        #   expected: list[doc],
        #   doc: list[page],
        #   page: list[feature]
        #
        # list[feature] is sorted by feature name (not value!)
        expected = [
            # invalid1.pdf
            [[None, None]],

            # single_page.pdf
            [['975f5049aac2d0b0e85e0083657182fd', 'My beautiful sample!']],

            # sub2/copy_single_page.pdf
            [['975f5049aac2d0b0e85e0083657182fd', 'My beautiful sample!']],

            # multi_page.pdf
            [['5f005131323536c524e0fffa7ab42d0f', 'First page'],
                ['fce06f79de9ca575212152873cb161ea', 'Second page'],
                ['6dacd3629627df0d99ebc5da97b310b2', 'Third page']],

            # sub1/copy_multi_page.pdf
            [['5f005131323536c524e0fffa7ab42d0f', 'First page'],
                ['fce06f79de9ca575212152873cb161ea', 'Second page'],
                ['6dacd3629627df0d99ebc5da97b310b2', 'Third page']],
        ]

        assert sorted(list_, key=sort) == sorted(expected, key=sort)


class TestParams:
    def test_features_as_list(self, expected_all):
        df = extract(SAMPLES_DIR, small=True, features=['text', 'image'])
        check_and_compare(df, expected_all)

    @pytest.mark.parametrize('excluded', [
        'text',
        'image',
    ])
    def test_exclude_feature(self, excluded, expected_all):
        features = PdfExtractTask.list_features()
        features.remove(excluded)

        df = extract(SAMPLES_DIR, small=True, features=features)

        columns = list(expected_all.columns)
        columns.remove(excluded)

        check_and_compare(df, expected_all[columns])

    def test_empty_feature(self, expected_all):
        df = extract(SAMPLES_DIR, small=True, features='')

        columns = list(PdfExtractTask.fixed_featues) + ['error_bool']
        check_and_compare(df, expected_all[columns])

    @pytest.mark.parametrize('size', (
        ('10x10'),
        ('10X10'),
        ('10x100'),
    ))
    def test_image_resize(self, size):
        df = extract(SAMPLES_DIR, image_size=size,
                     small=True, features='image')

        img_bytes = df['image'].dropna().iloc[0]
        img = Image.open(BytesIO(img_bytes))

        size = size.lower()
        width, height = map(int, size.split('x'))

        assert img.size == (width, height)

    @pytest.mark.parametrize('format_', (
        'jpeg',
        'png',
    ))
    def test_image_format(self, format_):
        df = extract(SAMPLES_DIR, image_format=format_,
                     small=True, features='image')

        img_bytes = df['image'].dropna().iloc[0]
        img = Image.open(BytesIO(img_bytes))

        assert img.format.upper() == format_.upper()

    @pytest.mark.parametrize('ocr_image_size,is_low', (
        (200, True),
        (2000, False),
    ))
    def test_low_ocr_image(self, expected_all, ocr_image_size, is_low):
        df = extract_text(
            SAMPLES_DIR, small=True, ocr=True,
            ocr_image_size=ocr_image_size, ocr_lang='eng'
        )

        df = df.dropna(subset=['text'])
        serie = df.iloc[0]

        expected = expected_all.dropna(subset=['text'])
        expected = expected[(expected.path == serie.path)
                            & (expected.page == serie.page)]

        expected_serie = expected.iloc[0]

        if is_low:
            assert serie.text.strip() != expected_serie.text.strip()
        else:
            assert serie.text.strip() == expected_serie.text.strip()

    def test_imagefrombytes(self, image, image_bytes):

        assert image_from_bytes(image_bytes) == image

    def test_imagetobytes(self, image, image_bytes):
        # png because jpeg change pixel values
        calculated = image_from_bytes(image_to_bytes(image, 'png'))

        assert (np.array(calculated) == np.array(image)).all()


class TestExtractionFromMemory:

    @pytest.mark.parametrize('small', (
        True,
        False,
    ))
    def test_passing_tasks(self, tmp_path, small):
        with open(SAMPLES_DIR / 'single_page1.pdf', 'rb') as f:
            pdf1_bin = f.read()

        with open(SAMPLES_DIR / 'multi_page1.pdf', 'rb') as f:
            pdf2_bin = f.read()

        tasks = [
            ('doc1.pdf', pdf1_bin),  # All pages
            ('2.pdf', pdf2_bin, 2),  # Just page 2
            ('pdf2.pdf', pdf2_bin, 3),  # Just page 3
        ]

        expected_dict = {
            'path': ['pdf2.pdf', '2.pdf', 'doc1.pdf'],
            'page': [3, 2, 1],
            'text': ['Third page', 'Second page', 'My beautiful sample!'],
            'error': [None, None, None],
        }
        expected = pd.DataFrame(expected_dict)

        if small:
            df = extract_text(tasks=tasks, small=small)
        else:
            result_path = tmp_path / 'result.parquet.gzip'
            extract_text(tasks, result_path)

            df = pd.read_parquet(result_path, engine=PARQUET_ENGINE)

        check_and_compare(df, expected, list(expected.columns))
