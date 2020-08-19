import copy
from hashlib import md5


class TestingDataFrame:
    def __init__(self, df):
        self._df = df
        self._df = self._df.fillna('').astype('str')

    def __repr__(self):
        columns = ', '.join(self._df.columns)
        values = str(self._df.values)

        return f'Columns: {columns}\n{values}'

    def _hash_images(self):
        if 'image' in self._df.columns:
            self._df['image'] = self._df['image'].apply(
                lambda c: md5(c.encode()).hexdigest() if c else None
            )

    def _pop_error_columns(self):
        for column in ('error', 'error_bool'):
            if column in self._df.columns:
                self._df.pop(column)

    def sort(self):
        columns = sorted(self._df.columns)

        self._df.sort_values(by=columns, inplace=True)
        self._df = self._df[columns]

    def check_errors(self, is_ocr):
        def check_error_msg(error):
            if error:
                assert 'Traceback' in error

                if 'text' in self._df.columns:
                    pdftotext_error_msg = 'poppler error creating document'
                    assert (pdftotext_error_msg in error) != is_ocr

        self._df['error'].apply(check_error_msg)

    def _assert_equal_errors(self, other):
        check_column = 'error'

        if check_column not in other._df.columns:
            self._df['error_bool'] = self._df.pop('error').apply(bool)
            check_column = 'error_bool'

        self_values = self._df[check_column].values
        other_values = other._df[check_column].values

        return (self_values == other_values).all()

    def assert_equal(self, expected):
        self_cp = copy.deepcopy(self)
        expected_cp = copy.deepcopy(expected)

        self_cp._assert_equal_errors(expected_cp)
        self_cp._pop_error_columns()
        expected_cp._pop_error_columns()

        # Improve visualization for debugging
        self_cp._hash_images()
        expected_cp._hash_images()

        self_cp.sort()
        expected_cp.sort()

        # Making debug easier
        assert list(self_cp._df.columns) == list(expected_cp._df.columns)
        assert self_cp._df.shape == expected_cp._df.shape

        assert (self_cp._df.values == expected_cp._df.values).all()

    def check_and_compare(self, expected, is_ocr=False):
        self.check_errors(is_ocr)

        self.assert_equal(expected)


def check_and_compare(df, expected, is_ocr=False):
    expected = TestingDataFrame(expected)
    df = TestingDataFrame(df)

    df.check_and_compare(expected, is_ocr)
