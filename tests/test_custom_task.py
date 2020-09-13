from pathlib import Path

import pyarrow as pa
from pdf2dataset import extract, feature, PdfExtractTask


class MyCustomTask(PdfExtractTask):

    @feature('bool_')
    def get_is_page_even(self):
        return self.page % 2 == 0

    @feature(is_helper=True)
    def get_doc_first_bytes(self):
        return self.file_bin[:10]

    @feature('list_', value_type=pa.string())
    def get_list(self):
        return ['E0', 'E1', 'My super string!']

    @feature('string', exceptions=[ValueError])
    def get_wrong(self):
        raise ValueError("There was a problem!")


def test_list_features():
    inherited_features = PdfExtractTask.list_features()
    custom_features = MyCustomTask.list_features()

    # 3 because I've defined this number of (not helpers) custom features
    expected_num_features = len(inherited_features) + 3
    assert expected_num_features == len(custom_features)

    assert set(inherited_features) < set(custom_features)

    assert set(['is_page_even', 'wrong', 'list']) < set(custom_features)


def test_list_helper_features():
    inherited_features = PdfExtractTask.list_helper_features()
    custom_features = MyCustomTask.list_helper_features()

    # 1 because I've defined one helpers custom feature
    expected_num_features = len(inherited_features) + 1
    assert expected_num_features == len(custom_features)

    assert set(inherited_features) < set(custom_features)

    assert set(['doc_first_bytes']) < set(custom_features)


def test_saving_to_disk(tmp_path):
    out_file = tmp_path / 'my_df.parquet.gzip'
    extract('tests/samples', out_file, task_class=MyCustomTask)

    assert Path(out_file).exists()


def test_columns_present():
    df = extract('tests/samples', small=True, task_class=MyCustomTask)
    assert set(MyCustomTask.list_features()) < set(df.columns)


def test_error_recorded():
    df = extract('tests/samples', small=True, task_class=MyCustomTask)
    error_feature, error_msg = 'wrong', 'There was a problem'

    assert error_msg in df.iloc[0].error
    assert f'{error_feature}:' in df.iloc[0].error
