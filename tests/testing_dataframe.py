class TestingDataFrame:
    def __init__(self, df):
        self._df = df
        self._df = self._df.fillna('').astype('str').astype(str)

    def __repr__(self):
        columns = ', '.join(self._df.columns)
        values = str(self._df.values)

        return f'Columns: {columns}\n{values}'

    def _pop_error_columns(self):
        for column in ('error', 'error_bool'):
            if column in self._df.columns:
                self._df.pop(column)

    def sort(self):
        columns = sorted(self._df.columns)

        self._df = self._df[columns]
        sorted_df = self._df.sort_values(by=columns)

        return TestingDataFrame(sorted_df)

    def check_errors(self, is_ocr):
        def check_error_msg(error):
            if error:
                assert 'Traceback' in error

                pdftotext_error_msg = 'poppler error creating document'
                assert (pdftotext_error_msg in error) != is_ocr

        self._df['error'].apply(check_error_msg)

    def _assert_equal_errors(self, other):
        check_column = 'error'

        if 'error' not in other._df.columns:
            self._df['error_bool'] = self._df.pop('error').apply(bool)
            check_column = 'error_bool'

        self_values = self._df[check_column].values
        other_values = other._df[check_column].values

        return (self_values == other_values).all()

    def assert_equal(self, other):
        self_copy = self.sort()
        other_copy = other.sort()

        self_copy._assert_equal_errors(other_copy)
        self_copy._pop_error_columns()
        other_copy._pop_error_columns()

        # Making debug easier
        assert sorted(self_copy._df.columns) == sorted(other_copy._df.columns)
        assert self_copy._df.shape == other_copy._df.shape

        assert (self_copy._df.values == other_copy._df.values).all()

    def check_and_compare(self, expected, is_ocr=False):
        self.check_errors(is_ocr)

        self.assert_equal(expected)


def check_and_compare(df, expected, is_ocr=False):
    expected = TestingDataFrame(expected)
    df = TestingDataFrame(df)

    df.check_and_compare(expected, is_ocr)
