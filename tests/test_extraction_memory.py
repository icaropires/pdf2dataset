import pytest
import pandas as pd
from pdf2dataset import extract_text

from .testing_dataframe import check_and_compare
from .conftest import SAMPLES_DIR, PARQUET_ENGINE


@pytest.mark.parametrize('small', (
    True,
    False,
))
def test_passing_tasks(tmp_path, small):
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
