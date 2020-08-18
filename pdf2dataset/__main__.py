import argparse
from . import TextExtraction


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
    parser.add_argument(
        '--features',
        type=str,
        default='text',
        help=(
            'Specify a comma separated list with the features you want to'
            " extract. 'path' and 'page' will always be added."
            " Example 1: --features=text,image"
            " Example 2: --features=all"
            ' Available features to add: text, image'
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
        '--lang',
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
        '--img-size',
        type=str,
        default=None,
        help=(
            'If adding image feature, image will be resized to this size.'
            " Provide two integers separated by 'x'."
            ' Example: --img-size 1000x1414'
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
        help='Redis password to use to connect with redis'
    )

    args = parser.parse_args()

    extraction = TextExtraction(**vars(args))
    extraction.apply()

    print(f"Results saved to '{extraction.results_file}'!")


if __name__ == '__main__':
    main()
