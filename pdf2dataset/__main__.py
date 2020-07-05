import ray
import argparse
from pathlib import Path
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
        'output_dir',
        type=str,
        help=('The folder to keep all the results, including log files and'
              ' intermediate files')
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=TextExtraction.default_workers,
        help='Workers to use in the pool'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        default='df.parquet.gzip',
        help='File to save a pickle with the results'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='por',
        help='Tesseract language'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_file = Path(args.results_file)
    max_workers = args.workers
    lang = args.lang

    ray.init()
    print()

    extraction = TextExtraction(
        input_dir,
        results_file,
        output_dir=output_dir,
        max_workers=max_workers,
        lang=lang
    )
    extraction.apply()

    print(f"Results saved to '{results_file}'!")


if __name__ == '__main__':
    main()
