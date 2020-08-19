import argparse

from . import Extraction
from .extraction_task import ExtractionTask


# TODO: use click
def main():
    parser = argparse.ArgumentParser(
        description='Extract text from all PDF files in a directory'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='The folder to lookup for PDF files recursively'
    )
    parser.add_argument(
        'results_file',
        type=str,
        default='df.parquet.gzip',
        help='File to save the resultant dataframe'
    )

    available_featues = ', '.join(ExtractionTask.list_features())
    parser.add_argument(
        '--features',
        type=str,
        default='all',
        help=(
            'Specify a comma separated list with the features you want to'
            " extract. 'path' and 'page' will always be added."
            f' Available features to add: {available_featues}'
            " Examples: '--features=text,image' or '--features=all'"
        )
    )

    parser.add_argument(
        '--tmp-dir',
        type=str,
        default='',
        help=('The folder to keep all the results, including log files and'
              ' intermediate files')
    )
    parser.add_argument(
        '--ocr-lang',
        type=str,
        default='por',
        help='Tesseract language'
    )
    parser.add_argument(
        '--ocr',
        type=bool,
        default=False,
        help="'pytesseract' if true, else 'pdftotext'. default: false"
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        help="Chunksize to use while processing pages, otherwise is calculated"
    )
    parser.add_argument(
        '--image-size',
        type=str,
        default=None,
        help=(
            'If adding image feature, image will be resized to this size.'
            " Provide two integers separated by 'x'."
            ' Example: --image-size 1000x1414'
        )
    )
    parser.add_argument(
        '--ocr-image-size',
        type=int,
        default=None,
        help=(
            'The height of the image OCR will be applied.'
            ' Width will be adjusted to keep the ratio.'
        )
    )
    parser.add_argument(
        '--image-format',
        type=str,
        default='jpeg',
        help=(
            'Format of the image generated from the PDF pages'
        )
    )

    # Ray
    parser.add_argument(
        '--num-cpus',
        type=int,
        help='Number of cpus to use'
    )
    parser.add_argument(
        '--address',
        type=str,
        help='Ray address to connect'
    )
    parser.add_argument(
        '--webui-host',
        type=str,
        default='*',
        help='Which IP ray webui will try to listen on'
    )
    parser.add_argument(
        '--redis-password',
        type=str,
        default='5241590000000000',  # Ray default
        help='Redis password to use to connect with ray'
    )

    args = parser.parse_args()

    extraction = Extraction(**vars(args))
    extraction.apply()

    print(f"Results saved to '{extraction.results_file}'!")


if __name__ == '__main__':
    main()
