import re
from pathlib import Path

import pytest
import pandas as pd
from pdf2dataset import TextExtraction, extract_text


@pytest.mark.parametrize('use_ocr', (
    True,
    False,
))
def test_general(tmp_path, use_ocr):
    result_path = tmp_path / 'result.parquet.gzip'

    extract_text('tests/samples', result_path, lang='eng', ocr=use_ocr)

    df = pd.read_parquet(result_path, engine='fastparquet')
    assert df.shape == (9, 4)

    rows = sorted([r[1:] for r in df.itertuples()])

    # Error as boolean just for testing
    expected = sorted([
        ('multi_page1.pdf', 1, 'First page', False),
        ('multi_page1.pdf', 2, 'Second page', False),
        ('multi_page1.pdf', 3, 'Third page', False),
        ('sub1/copy_multi_page1.pdf', 1, 'First page', False),
        ('sub1/copy_multi_page1.pdf', 2, 'Second page', False),
        ('sub1/copy_multi_page1.pdf', 3, 'Third page', False),
        ('single_page1.pdf', 1, 'My beautiful sample!', False),
        ('sub2/copy_single_page1.pdf', 1, 'My beautiful sample!', False),
        ('invalid1.pdf', -1, '', True)
    ])

    for idx, r in enumerate(rows):
        *other, error = r
        assert tuple(other) == expected[idx][:-1]

        has_error = expected[idx][-1]
        assert bool(error) == has_error

        if has_error:
            assert 'Traceback' in error

            pdftotext_error_msg = 'poppler error creating document'
            assert (pdftotext_error_msg in error) != use_ocr


@pytest.mark.parametrize('use_ocr', (
    True,
    False,
))
def test_tmpdir(tmp_path, tmpdir, use_ocr):
    result_path = tmp_path / 'result.parquet.gzip'
    tmp_dir = Path(tmpdir.mkdir('tmp'))

    extract_text('tests/samples', result_path, tmp_dir=tmp_dir,
                 lang='eng', ocr=use_ocr)

    expected_files = [
        ('multi_page1_1.txt'),
        ('multi_page1_2.txt'),
        ('multi_page1_3.txt'),
        ('sub1'),
        ('sub1/copy_multi_page1_1.txt'),
        ('sub1/copy_multi_page1_2.txt'),
        ('sub1/copy_multi_page1_3.txt'),
        ('single_page1_1.txt'),
        ('sub2'),
        ('sub2/copy_single_page1_1.txt'),
        ('invalid1_-1_error.log')  # -1 is the whole document
    ]

    tmp_files = list(tmp_dir.rglob('*'))
    assert len(tmp_files) == 11

    for f in tmp_files:
        f = str(f.relative_to(tmp_dir))
        assert f in expected_files


@pytest.mark.parametrize('path,expected', (
    ('multi_page1_1.txt', {'path': 'multi_page1', 'page': '1'}),
    ('multi_page1_2.txt', {'path': 'multi_page1', 'page': '2'}),
    ('multi_page1_3.txt', {'path': 'multi_page1', 'page': '3'}),
    ('multi_page1_10.txt', {'path': 'multi_page1', 'page': '10'}),
    ('multi_page1_101.txt', {'path': 'multi_page1', 'page': '101'}),
    ('sub1/multi_page1_100.txt', {'path': 'sub1/multi_page1', 'page': '100'}),
    ('s1/s2/doc_1000.txt', {'path': 's1/s2/doc', 'page': '1000'}),
    ('single_page1_1.txt', {'path': 'single_page1', 'page': '1'}),
    ('invalid1_doc_error.log', {'path': 'invalid1', 'page': 'doc'}),
    ('invalid1_-10_error.log', {'path': 'invalid1', 'page': '-10'}),
    ('invalid1_10_error.log', {'path': 'invalid1', 'page': '10'}),
))
def test_regex_extract(path, expected):
    result = re.match(TextExtraction._path_pat, path)
    assert result.groupdict() == expected
