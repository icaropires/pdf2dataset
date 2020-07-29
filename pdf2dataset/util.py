from pathlib import Path

from .extract import TextExtraction
from .extract_not_dir import TextExtractionNotDir


def extract_text(*args, return_list=False, **kwargs):
    input_dir = kwargs.get('input_dir')

    if len(args) and isinstance(args[0], (str, Path)) or input_dir:
        extraction = TextExtraction(*args, **kwargs)
    else:
        extraction = TextExtractionNotDir(*args, **kwargs)

    df = extraction.apply()

    if return_list:
        return df.groupby('path')['text'].agg(list).to_list()

    return df
