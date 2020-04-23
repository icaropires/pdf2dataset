#!/bin/env python3

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import count

import cv2
import pytesseract
from pdf2image import convert_from_path


PDFS_PATH = Path('/mnt/nas/tjrr_dados/pdfs')


def get_page_img(path, page_number):
    img = convert_from_path(path, first_page=page_number,
                            last_page=page_number, fmt='jpeg')
    return img[0]


def ocr_image(img):
    tsh = np.array(img.convert('L'))
    tsh = cv2.adaptiveThreshold(tsh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 97, 50)

    erd = cv2.erode(
        tsh,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        iterations=1
    )

    text_page = pytesseract.image_to_string(erd, lang='por')

    return text_page


def process_file(path):
    def write_result(name, text):
        with open(out_path, 'w') as f:
            f.write(text)

    try:
        img = get_page_img(path, 1)
    except Exception as e:
        print(f'{path.name} skipped: {e}', end='\n\n')

        out_path = str(path).replace('.pdf', '.txt')
        write_result(out_path, 'ST_SKIPPED')
        return

    for i in count(1):
        try:
            img = get_page_img(path, i)
            text = ocr_image(img)

            out_path = str(path).replace('.pdf', f'_{i}.txt')
            write_result(out_path, text)
        except IndexError:
            break
        except Exception as e:
            print(f'{path.name}_{i}: {e}', end='\n\n')
            text = ''


print('Looking for PDF files...')
pdf_files = list(PDFS_PATH.glob('**/*.pdf'))
print(f'Found {len(pdf_files)} documents! Starting processing...')

with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
    with tqdm(total=len(pdf_files)) as pbar:
        futures = []
        for f in pdf_files:
            future = executor.submit(process_file, f)
            future.add_done_callback(lambda p: pbar.update())
            futures.append(future)

        for future in futures:
            future.result()
