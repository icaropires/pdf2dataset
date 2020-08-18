import re
from pathlib import Path

import pytest
import pandas as pd
from pdf2dataset import TextExtraction, extract_text

from .testing_dataframe import check_and_compare


SAMPLES_DIR = 'tests/samples'
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


class TestExtractionCore:

    @pytest.mark.parametrize('is_ocr', (
        True,
        False,
    ))
    def test_extraction_big(self, tmp_path, is_ocr, expected_all):
        result_path = tmp_path / 'result.parquet.gzip'

        extract_text(SAMPLES_DIR, result_path,
                     lang='eng', ocr=is_ocr, features='all')

        df = pd.read_parquet(result_path, engine=PARQUET_ENGINE)
        check_and_compare(df, expected_all, is_ocr=is_ocr)

    def test_tmpdir(self, tmp_path, tmpdir):
        result_path = tmp_path / 'result.parquet.gzip'
        tmp_dir = Path(tmpdir.mkdir('tmp'))

        extract_text(SAMPLES_DIR, result_path, tmp_dir=tmp_dir, lang='eng')

        features = ['text', 'error']
        folders = ['sub1', 'sub2']
        prefix_pages = (
                ('multi_page1', [1, 2, 3]),
                ('single_page1', [1]),
                ('sub1/copy_multi_page1', [1, 2, 3]),
                ('sub2/copy_single_page1', [1]),
                ('invalid1', [-1]),
                )

        expected_files = [
                f'{prefix}_{feature}_{page}.txt'
                for prefix, pages in prefix_pages
                for page in pages
                for feature in features
        ]

        expected_files += folders

        tmp_files = list(tmp_dir.rglob('*'))
        tmp_files = [str(f.relative_to(tmp_dir)) for f in tmp_files]

        assert sorted(tmp_files) == sorted(expected_files)

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
        result = re.match(TextExtraction._path_pat, path)
        assert result.groupdict() == expected


class TestExtractionSmall:
    @pytest.mark.parametrize('is_ocr', (
        True,
        False,
    ))
    def test_extraction_small(self, is_ocr, expected_all):
        df = extract_text(SAMPLES_DIR, small=True,
                          lang='eng', ocr=is_ocr, features='all')

        check_and_compare(df, expected_all, is_ocr=is_ocr)

    def test_return_list(self):
        texts_list = extract_text(SAMPLES_DIR,
                                  return_list=True, lang='eng')

        texts_list = sorted(texts_list, key=lambda x: len(x))

        expected = [
            # invalid1.pdf
            [None],

            # single_page.pdf
            ['My beautiful sample!'],

            # sub2/copy_single_page.pdf
            ['My beautiful sample!'],

            # multi_page.pdf
            ['First page', 'Second page', 'Third page'],

            # sub1/copy_multi_page.pdf
            ['First page', 'Second page', 'Third page'],
        ]

        assert texts_list == expected


class TestFeaturesParams:
    def test_no_text(self, expected_all):
        df = extract_text(SAMPLES_DIR, small=True,
                          lang='eng', features='image')

        columns = [c for c in expected_all.columns if c != 'text']
        check_and_compare(df, expected_all[columns])

    def test_no_image(self, expected_all):
        df = extract_text(SAMPLES_DIR, small=True,
                          lang='eng', features='text')

        columns = [c for c in expected_all.columns if c != 'image']
        check_and_compare(df, expected_all[columns])

    def test_none(self, expected_all):
        df = extract_text(SAMPLES_DIR, small=True, lang='eng', features='')

        columns = ['path', 'page', 'error_bool']
        check_and_compare(df, expected_all[columns])


class TestExtractionNotDir:

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
