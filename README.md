# pdf2dataset

![pdf2dataset](https://github.com/icaropires/pdf2dataset/workflows/pdf2dataset/badge.svg)

Converts a whole subdirectory with big volume of PDF documents to a dataset (pandas DataFrame) with the columns: path x page x text x error


## Highlights

* Conversion of a whole subdirectory with PDFs documents into a pandas DataFrame
* Support for parallel and distributed computing through [ray](https://github.com/ray-project/ray)
* Incremental writing of resulting DataFrame, to save memory
* Ability to keep processing progress and resume from it
* Error tracking of faulty documents
* Use OCR for extracting text through [pytesseract](https://github.com/madmaze/pytesseract) and [pdf2image](https://github.com/Belval/pdf2image)
* Custom behaviour through parameters (number of CPUs, text language, etc)


## Install

### Install Dependencies

#### Ubuntu (or debians)

``` bash
$ sudo apt update
$ sudo apt install -y poppler-utils tesseract-ocr-por  # "-por" for portuguese, use your language
```

### Install pdf2dataset

#### For usage

``` bash
$ pip3 install pdf2dataset --user # Please, isolate the environment
```


#### For development

``` bash
# First, clone repository and cd into it
$ poetry install
```


## Usage

### Simple

``` bash
# Reads all PDFs from my_pdfs_folder and saves the resultant dataframe to my_df.parquet.gzip
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip
```

### Keeping progress

``` bash
# Keep progress in tmp folder, so can resume processing in case of any error or interruption
# To resume, just use the same --tmp-dir folder
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip --tmp-dir my_progress
```
### Results File

The resulting "file" is a parquet hive written with [fastparquet](https://github.com/dask/fastparquet), it can be
easily read with pandas or dask:

``` python
>>> import pandas as pd
>>> df = pd.read_parquet('my_df.parquet.gzip')
>>> df
                             path  page                  text                                              error
index                                                                                                           
0                single_page1.pdf     1  My beautiful sample!                                                   
1       sub1/copy_multi_page1.pdf     2           Second page                                                   
2      sub2/copy_single_page1.pdf     1  My beautiful sample!                                                   
3       sub1/copy_multi_page1.pdf     3            Third page                                                   
4                 multi_page1.pdf     1            First page                                                   
5                 multi_page1.pdf     3            Third page                                                   
6       sub1/copy_multi_page1.pdf     1            First page                                                   
7                 multi_page1.pdf     2           Second page                                                   
0                    invalid1.pdf    -1                        Traceback (most recent call last):\n  File "/h...
```

There is no guarantee about the uniqueness or sequence of the `index`, you might need to create a new index with
the whole data in memory.

The `-1` page number means that was not possible of even openning the document.

### Help

``` bash
$ pdf2dataset -h
usage: pdf2dataset [-h] [--tmp-dir TMP_DIR] [--lang LANG]
                   [--num-cpus NUM_CPUS] [--address ADDRESS]
                   [--webui-host WEBUI_HOST] [--redis-password REDIS_PASSWORD]
                   input_dir results_file

Extract text from all PDF files in a directory

positional arguments:
  input_dir             The folder to lookup for PDF files recursively
  results_file          File to save the resultant dataframe

optional arguments:
  -h, --help            show this help message and exit
  --tmp-dir TMP_DIR     The folder to keep all the results, including log
                        files and intermediate files
  --lang LANG           Tesseract language
  --num-cpus NUM_CPUS   Number of cpus to use
  --address ADDRESS     Ray address to connect
  --webui-host WEBUI_HOST
                        Which port ray webui to listen
  --redis-password REDIS_PASSWORD
                        Redis password to use to connect with redis
```


## Troubleshooting

1. **Troubles with high memory usage**

You can try to decrease the number of CPUs in use, reducing the level of
parallelism, test with `--num-cpus 1` flag and then increasy according to your hardware.
