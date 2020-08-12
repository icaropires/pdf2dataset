import re
from pathlib import Path

import pytest
import pandas as pd
from pdf2dataset import TextExtraction, extract_text


@pytest.fixture
def expected_all():
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
    all_ = {n: r for n, r in zip(names, zip(*rows))}

    return pd.DataFrame(all_)


def check_df(df, use_ocr, expected=None):
    'Check dataframe based on samples folder'

    def check_error_msg(error):
        if error is not None:
            assert 'Traceback' in error

            pdftotext_error_msg = 'poppler error creating document'
            assert (pdftotext_error_msg in error) != use_ocr

    if expected is None:
        expected = expected_all

    df['error'].apply(check_error_msg)
    df['error_bool'] = df.pop('error').apply(bool)

    columns = list(df.columns)
    df.sort_values(by=columns, inplace=True)
    expected.sort_values(by=columns, inplace=True)

    assert (df.values == expected.values).all()


class TestExtraction:

    @pytest.mark.parametrize('use_ocr', (
        True,
        False,
    ))
    def test_extraction_big(self, tmp_path, use_ocr, expected_all):
        result_path = tmp_path / 'result.parquet.gzip'

        extract_text('tests/samples', result_path, lang='eng', ocr=use_ocr)

        df = pd.read_parquet(result_path, engine='fastparquet')
        check_df(df, use_ocr, expected_all)

    @pytest.mark.parametrize('use_ocr', (
        True,
        False,
    ))
    def test_extraction_small(self, tmp_path, use_ocr, expected_all):
        df = extract_text('tests/samples', small=True, lang='eng', ocr=use_ocr)
        check_df(df, use_ocr, expected_all)

    def test_return_list(self):
        texts_list = extract_text('tests/samples',
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

    def test_tmpdir(self, tmp_path, tmpdir):
        result_path = tmp_path / 'result.parquet.gzip'
        tmp_dir = Path(tmpdir.mkdir('tmp'))

        extract_text('tests/samples', result_path, tmp_dir=tmp_dir, lang='eng')

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
        ('multi_page1_doc_101.txt',
            {'path': 'multi_page1', 'page': '101', 'feature': 'doc'}),
        ('s1/s2/doc_img_1000.txt',
            {'path': 's1/s2/doc', 'page': '1000', 'feature': 'img'}),
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


class TestExtractionNotDir:

    @pytest.mark.parametrize('small', (
        True,
        False,
    ))
    def test_pass_tasks(self, tmp_path, small):
        with open('tests/samples/single_page1.pdf', 'rb') as f:
            pdf1_bin = f.read()

        with open('tests/samples/multi_page1.pdf', 'rb') as f:
            pdf2_bin = f.read()

        tasks = [
            ('doc1.pdf', pdf1_bin),  # All pages
            ('2.pdf', pdf2_bin, 2),  # Just page 2
            ('pdf2.pdf', pdf2_bin, 3),  # Just page 3
        ]

        expected = {
            'path': ['pdf2.pdf', '2.pdf', 'doc1.pdf'],
            'page': [3, 2, 1],
            'text': ['Third page', 'Second page', 'My beautiful sample!'],
            'error_bool': [False, False, False],
        }
        expected = pd.DataFrame(expected)

        if small:
            df = extract_text(tasks=tasks, small=small)
        else:
            result_path = tmp_path / 'result.parquet.gzip'
            extract_text(tasks, result_path)

            df = pd.read_parquet(result_path, engine='fastparquet')

        check_df(df, False, expected)
