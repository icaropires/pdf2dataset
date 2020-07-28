from pathlib import Path

from .extract import TextExtraction
from .extract_not_dir import TextExtractionNotDir


def extract_text(*args, **kwargs):
    input_dir = kwargs.get('input_dir')

    if len(args) and isinstance(args[0], (str, Path)) or input_dir:
        extraction = TextExtraction(*args, **kwargs)
    else:
        extraction = TextExtractionNotDir(*args, **kwargs)

    return extraction.apply()
