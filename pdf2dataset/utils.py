from pathlib import Path

from .extraction import Extraction
from .extraction_memory import ExtractionFromMemory


def _df_to_list(df, extraction):
    fixed_featues = list(extraction.task_class.fixed_featues)

    df.set_index(fixed_featues, inplace=True)
    df.sort_index(inplace=True)
    df.pop('error')  # ignoring errors when returning list

    df = df[sorted(df.columns)]
    df['features'] = df.apply(lambda row: row.to_list(), axis='columns')

    groups = df.reset_index().groupby('path')['features']

    return groups.agg(list).to_list()


def extract(*args, return_list=False, **kwargs):
    input_dir = kwargs.get('input_dir')

    kwargs['small'] = True if return_list else kwargs.get('small', False)

    if len(args) and isinstance(args[0], (str, Path)) or input_dir:
        extraction = Extraction(*args, **kwargs)
    else:
        extraction = ExtractionFromMemory(*args, **kwargs)

    # None if small=False
    df = extraction.apply()

    if return_list:
        return _df_to_list(df, extraction)

    return df


def extract_text(*args, **kwargs):
    kwargs['features'] = 'text'
    return extract(*args, **kwargs)
