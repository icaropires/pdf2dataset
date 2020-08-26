import sys
import inspect

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
    def is_tasks(input_):
        return (
            kwargs.get('tasks')
            or (isinstance(input_, (list, tuple))
                and input_ and isinstance(input_[0], (list, tuple)))
        )

    args = list(args)
    input_ = args.pop(0) if args else None

    kwargs['small'] = True if return_list else kwargs.get('small', False)

    if inspect.isgenerator(input_):
        raise ValueError('Generator as input is not currently supported!')

    if is_tasks(input_):
        tasks = kwargs.pop('tasks', input_)
        extraction = ExtractionFromMemory(tasks, *args, **kwargs)
    else:
        if isinstance(input_, (list, tuple)):
            kwargs['files_list'] = input_
            input_ = ''

        extraction = Extraction(input_, *args, **kwargs)

    df = extraction.apply()  # df is None if small equals False

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
