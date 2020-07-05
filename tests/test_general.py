import pandas as pd
from pdf2dataset import TextExtraction


def test_general(tmp_path):
    result_path = tmp_path / 'result.parquet.gzip'
    output_dir = '/tmp/pdf2dataset'

    extraction = TextExtraction(
        'tests/samples',
        result_path,
        output_dir=output_dir,
        lang='eng'
    )
    extraction.apply()

    df = pd.read_parquet(result_path, engine='fastparquet')
    assert df.shape == (5, 3)

    df = df.set_index('path')
    texts_result = df['text'].to_dict().items()

    texts_expected = [
        (f'{output_dir}/multi_page1_1.txt', 'First page'),
        (f'{output_dir}/multi_page1_2.txt', 'Second page'),
        (f'{output_dir}/multi_page1_3.txt', 'Third page'),
        (f'{output_dir}/single_page1_1.txt', 'My beautiful sample!'),
        (f'{output_dir}/invalid1_doc_error.log', '')
    ]

    for te in texts_expected:
        assert te in texts_result

    errors_result = df['error'].to_dict()

    errors_expected = [
        (f'{output_dir}/multi_page1_1.txt', False),
        (f'{output_dir}/multi_page1_2.txt', False),
        (f'{output_dir}/multi_page1_3.txt', False),
        (f'{output_dir}/single_page1_1.txt', False),
        (f'{output_dir}/invalid1_doc_error.log', True)
    ]

    for doc, has_error in errors_expected:
        error_msg = errors_result[doc]
        assert bool(error_msg) == has_error

        if has_error:
            assert 'Traceback' in error_msg
