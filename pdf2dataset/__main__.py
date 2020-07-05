import ray
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
        '--tmp-dir',
        type=str,
        help=('The folder to keep all the results, including log files and'
              ' intermediate files')
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='por',
        help='Tesseract language'
    )

    args = parser.parse_args()

    ray.init()
    print()

    extraction = TextExtraction(**vars(args))
    extraction.apply()

    print(f"Results saved to '{extraction.results_file}'!")


if __name__ == '__main__':
    main()
