#!/bin/env python3

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import count
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm


# To pytesseract use only one core per worker
os.environ['OMP_THREAD_LIMIT'] = '1'


def get_page_img(path, page_number):
    img = convert_from_path(
        path,
        first_page=page_number,
        last_page=page_number,
        fmt='jpeg'
    )

    return img[0]


def ocr_image(img):
    tsh = np.array(img.convert('L'))
    tsh = cv2.adaptiveThreshold(
        tsh, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        97, 50
    )

    erd = cv2.erode(
        tsh,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        iterations=1
    )

    return pytesseract.image_to_string(erd, lang='por')


def get_pdf_files(input_dir):
    print('Looking for PDF files...')
    pdf_files = list(input_dir.glob('**/*.pdf'))
    print(f'Found {len(pdf_files)} documents! Starting processing...')

    return pdf_files


# TODO: Skip parsed files
def process_file(path, output_dir):
    '''
    Will replicate the directory structure of PDF files and save the results
    for each file in the corresponding position in the new structure
    '''
    error_suffix = '_error.log'

    out_path = output_dir / path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = out_path.with_name(out_path.stem + error_suffix)

    # Tests if can parse PDF
    try:
        img = get_page_img(path, 1)
    except Exception as e:
        with error_path.open('w') as f:
            f.write(f'{e}\n\n')

        return

    errors = []
    for page_num in count(1):
        try:
            img = get_page_img(path, page_num)
            text = ocr_image(img)

            name = out_path.stem + f'_{page_num}.txt'
            result_path = out_path.with_name(name)
            with result_path.open('w') as f:
                f.write(text)
        except IndexError:
            # All pages processed
            break
        except Exception as e:
            errors.append(f'Page {page_num}: {e}\n\n')

    if errors:
        with error_path.open('w') as f:
            f.write(''.join(errors))


if __name__ == '__main__':
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
        help='''The folder to keep all the results, including log files and
intermediate files'''
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pdf_files = get_pdf_files(input_dir)

    max_workers = min(32, os.cpu_count() + 4)
    print(f'PDFs directory: {input_dir}'
          f'\nOutput directory: {output_dir}'
          f'\nUsing {max_workers} workers', end='\n\n')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(pdf_files), unit='docs') as pbar:

            def submit(path):
                future = executor.submit(process_file, path, output_dir)
                future.add_done_callback(lambda _: pbar.update())

                return future

            for future in [submit(f) for f in pdf_files]:
                future.result()
