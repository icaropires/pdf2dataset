import sys
from pathlib import Path

from .extraction import Extraction
from .extraction_memory import ExtractionFromMemory
from .pdf_extract_task import PdfExtractTask, Image


def image_from_bytes(image_bytes):
    image = Image.from_bytes(image_bytes)
    return image.pil_image


def image_to_bytes(pil_image, image_format='jpeg'):
    image = Image(pil_image, image_format)
    return image.to_bytes()


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
    def is_path(param):
        return isinstance(param, (str, Path))

    kwargs['small'] = True if return_list else kwargs.get('small', False)

    if args and is_path(args[0]) or kwargs.get('input_dir'):
        extraction = Extraction(*args, **kwargs)
    else:
        extraction = ExtractionFromMemory(*args, **kwargs)

    # None if small=False
    df = extraction.apply()

    if return_list:
        return _df_to_list(df, extraction)

    return df


def gen_helpers(task_class=PdfExtractTask):
    def helpers_factory(feature):
        def helper(*args, **kwargs):
            kwargs['features'] = (feature if feature not in
                                  task_class.fixed_featues else '')

            return extract(*args, **kwargs)

        return helper

    def gen_helper(feature):
        method_name = f'extract_{feature}'
        setattr(sys.modules[__name__], method_name, helpers_factory(feature))

        return method_name

    features = task_class.list_features()
    return [gen_helper(feature) for feature in features]


HELPERS = gen_helpers()
