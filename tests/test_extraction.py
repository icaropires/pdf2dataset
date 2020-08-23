from io import BytesIO
import re
from pathlib import Path
from hashlib import md5

import pytest
import pandas as pd
import numpy as np
from PIL import Image

from pdf2dataset import (
    Extraction,
    PdfExtractTask,
    extract,
    extract_text,
    image_to_bytes,
    image_from_bytes,
)

from .testing_dataframe import check_and_compare


SAMPLES_DIR = 'tests/samples'
TEST_IMAGE = Path(SAMPLES_DIR) / 'single_page1_1.jpeg'
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

        extract(SAMPLES_DIR, result_path, chunk_df_size=1, features='all')

        # Small 'chunk_df_size' to append to result multiple times
        df = pd.read_parquet(result_path, engine=PARQUET_ENGINE)

        check_and_compare(df, expected_all)

    # TODO: Reenable feature
    # def test_tmpdir(self, tmp_path, tmpdir):
    #     result_path = tmp_path / 'result.parquet.gzip'
    #     tmp_dir = Path(tmpdir.mkdir('tmp'))

    #     extract(SAMPLES_DIR, result_path, tmp_dir=tmp_dir)

    #     features = ['text', 'error']
    #     folders = ['sub1', 'sub2']
    #     prefix_pages = (
    #         ('multi_page1', [1, 2, 3]),
    #         ('single_page1', [1]),
    #         ('sub1/copy_multi_page1', [1, 2, 3]),
    #         ('sub2/copy_single_page1', [1]),
    #         ('invalid1', [-1]),
    #     )

    #     expected_files = [
    #             f'{prefix}_{feature}_{page}.txt'
    #             for prefix, pages in prefix_pages
    #             for page in pages
    #             for feature in features
    #     ]

    #     expected_files += folders

    #     tmp_files = list(tmp_dir.rglob('*'))
    #     tmp_files = [str(f.relative_to(tmp_dir)) for f in tmp_files]

    #     assert sorted(tmp_files) == sorted(expected_files)

    @pytest.mark.parametrize('path,expected', (
        ('multi_page1_text_1.txt',
            {'path': 'multi_page1', 'page': '1', 'feature': 'text'}),
        ('multi_page1_text_2.txt',
            {'path': 'multi_page1', 'page': '2', 'feature': 'text'}),
        ('multi_page1_text_3.txt',
            {'path': 'multi_page1', 'page': '3', 'feature': 'text'}),
        ('multi_page1_text_10.txt',
            {'path': 'multi_page1', 'page': '10', 'feature': 'text'}),
        ('multi_page1_path_101.txt',
            {'path': 'multi_page1', 'page': '101', 'feature': 'path'}),
        ('s1/s2/doc_image_1000.txt',
            {'path': 's1/s2/doc', 'page': '1000', 'feature': 'image'}),
        ('single_page1_my-feature_1.txt',
            {'path': 'single_page1', 'page': '1', 'feature': 'my-feature'}),
        ('invalid1_error_-10.txt',
            {'path': 'invalid1', 'page': '-10', 'feature': 'error'}),
        ('invalid1_error_10.txt',
            {'path': 'invalid1', 'page': '10', 'feature': 'error'}),
    ))
    def test_path_pattern(self, path, expected):
        result = re.match(Extraction._path_pat, path)
        assert result.groupdict() == expected


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
    def test_no_text(self, expected_all):
        available_features = PdfExtractTask.list_features()
        features = list(set(available_features) - set(['text']))

        df = extract(SAMPLES_DIR, small=True, features=features)

        columns = [c for c in expected_all.columns if c != 'text']
        check_and_compare(df, expected_all[columns])

    def test_no_image(self, expected_all):
        available_features = PdfExtractTask.list_features()
        features = list(set(available_features) - set(['image']))

        df = extract(SAMPLES_DIR, small=True, features=features)

        columns = [c for c in expected_all.columns if c != 'image']
        check_and_compare(df, expected_all[columns])

    def test_features_as_list(self, expected_all):
        df = extract(SAMPLES_DIR, small=True, features=['text', 'image'])
        check_and_compare(df, expected_all)

    def test_none(self, expected_all):
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
        with open('tests/samples/single_page1.pdf', 'rb') as f:
            pdf1_bin = f.read()

        with open('tests/samples/multi_page1.pdf', 'rb') as f:
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
