import re

import pytest
import pandas as pd
from pdf2dataset import TextExtraction


def test_general(tmp_path):
    result_path = tmp_path / 'result.parquet.gzip'

    extraction = TextExtraction('tests/samples', result_path, lang='eng')
    extraction.apply()

    df = pd.read_parquet(result_path, engine='fastparquet')
    assert df.shape == (5, 3)

    df = df.set_index('path')
    texts_result = df['text'].to_dict().items()

    texts_expected = [
        ('multi_page1_1.txt', 'First page'),
        ('multi_page1_2.txt', 'Second page'),
        ('multi_page1_3.txt', 'Third page'),
        ('single_page1_1.txt', 'My beautiful sample!'),
        ('invalid1_doc_error.log', '')
    ]

    for te in texts_expected:
        assert te in texts_result

    errors_result = df['error'].to_dict()

    errors_expected = [
        ('multi_page1_1.txt', False),
        ('multi_page1_2.txt', False),
        ('multi_page1_3.txt', False),
        ('single_page1_1.txt', False),
        ('invalid1_doc_error.log', True)
    ]

    for doc, has_error in errors_expected:
        error_msg = errors_result[doc]
        assert bool(error_msg) == has_error

        if has_error:
            assert 'Traceback' in error_msg


def test_tmpdir(tmp_path, tmpdir):
    result_path = tmp_path / 'result.parquet.gzip'
    tmp_dir = tmpdir.mkdir('tmp')

    extraction = TextExtraction('tests/samples', result_path,
                                tmp_dir=tmp_dir, lang='eng')
    extraction.apply()

    expected_files = [
        ('multi_page1_1.txt'),
        ('multi_page1_2.txt'),
        ('multi_page1_3.txt'),
        ('single_page1_1.txt'),
        ('invalid1_doc_error.log')
    ]

    tmp_files = tmp_dir.listdir()
    assert len(tmp_files) == 5

    for f in tmp_files:
        assert f.basename in expected_files


@pytest.mark.parametrize('path,expected', (
    ('multi_page1_1.txt', {'path': 'multi_page1', 'page': '1'}),
    ('multi_page1_2.txt', {'path': 'multi_page1', 'page': '2'}),
    ('multi_page1_3.txt', {'path': 'multi_page1', 'page': '3'}),
    ('multi_page1_10.txt', {'path': 'multi_page1', 'page': '10'}),
    ('multi_page1_101.txt', {'path': 'multi_page1', 'page': '101'}),
    ('single_page1_1.txt', {'path': 'single_page1', 'page': '1'}),
    ('invalid1_doc_error.log', {'path': 'invalid1', 'page': 'doc'}),
    ('invalid1_-1_error.log', {'path': 'invalid1', 'page': '-1'}),
    ('invalid1_10_error.log', {'path': 'invalid1', 'page': '10'}),
))
def test_regex_extract(path, expected):
    result = re.match(TextExtraction._filepath_pat, path)
    assert result.groupdict() == expected
