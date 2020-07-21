# pdf2dataset

![pdf2dataset](https://github.com/icaropires/pdf2dataset/workflows/pdf2dataset/badge.svg)

Converts a whole subdirectory with big a volume of PDF documents to a dataset (pandas DataFrame) with the columns: path x page x text x error


## Highlights

* Conversion of a whole subdirectory with PDFs documents into a pandas DataFrame
* Support for parallel and distributed computing through [ray](https://github.com/ray-project/ray)
* Incremental writing of resulting DataFrame, to save memory
* Ability to save processing progress and resume from it
* Error tracking of faulty documents
* Ability to extracting text through [pdftotext](https://github.com/jalan/pdftotext)
* Ability to use OCR for extracting text through [pytesseract](https://github.com/madmaze/pytesseract) and [pdf2image](https://github.com/Belval/pdf2image)
* Custom behavior through parameters (number of CPUs, text language, etc)


## Installation

### Install Dependencies

#### Fedora

``` bash
# "-por" for portuguese, use the documents language
$ sudo dnf install -y poppler-utils pkgconfig poppler-cpp-devel python3-devel tesseract-langpack-por
```

#### Ubuntu (or debians)

``` bash
$ sudo apt update

# "-por" for portuguese, use the documents language
$ sudo apt install -y poppler-utils build-essential libpoppler-cpp-dev pkg-config python3-dev tesseract-ocr-por
```

### Install pdf2dataset

#### For usage

``` bash
$ pip3 install pdf2dataset --user  # Please, isolate the environment
```

#### For development

``` bash
# First, clone repository and cd into it
$ poetry install
```


## Usage

### Simple - CLI

``` bash
# Reads all PDFs from my_pdfs_folder and saves the resultant dataframe to my_df.parquet.gzip
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip  # Most basic
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip --num-cpus 1  # Reduce parallelism to the maximum
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip --ocr true  # For scanned PDFs
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip --ocr true --lang eng  # Scanned documents with english text
```

### Save Processing Progress - CLI

It's possible to save the progress to a temporary folder and resume from the saved state in case of
any error or interruption. To resume the processing, just use the `--tmp-dir [directory]` flag:

``` bash
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip --tmp-dir my_progress
```

The indicated temporary directory can also be used for debugging purposes and **is not** deleted
automatically, so delete it when desired. 


### Using as a library

The `extract_text` function can be used analogously to the CLI:

``` python
from pdf2dataset import extract_text

extract_text('my_pdfs_folder', 'my_df.parquet.gzip', tmp_dir='my_progress')
```

#### Small

One feature not available to the CLI is the custom behavior for handling small volumes of data (small can
be understood as that the extraction won't run for hours or days and locally).

The complete list of differences are:

* Don't save processing progress
* Distributed processing not supported
* Don't write dataframe to disk
* Returns the dataframe

##### Example:
``` python
from pdf2dataset import extract_text

df = extract_text('my_pdfs_folder', small=True)
# ...
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

The `-1` page number means that was not possible of even opening the document.

### Run on a Cluster

#### Setup the Cluster

Follow ray documentation for [manual](https://docs.ray.io/en/latest/using-ray-on-a-cluster.html?setup#manual-cluster-setup) or [automatic](https://docs.ray.io/en/latest/autoscaling.html?setup#automatic-cluster-setup)
setup.

#### Run it

To go distributed you can run it just like local, but using the `--address` and `--redis-password` flags to point to your cluster ([More information](https://docs.ray.io/en/latest/multiprocessing.html)).

With version >= 0.2.0, only the head node needs to have access to the documents in disk.

### Help

```
$ pdf2dataset -h
usage: pdf2dataset [-h] [--tmp-dir TMP_DIR] [--lang LANG] [--ocr OCR]
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
  --ocr OCR             'pytesseract' if true, else 'pdftotext'. default:
                        false
  --num-cpus NUM_CPUS   Number of cpus to use
  --address ADDRESS     Ray address to connect
  --webui-host WEBUI_HOST
                        Which port ray webui to listen
  --redis-password REDIS_PASSWORD
                        Redis password to use to connect with redis
```


## Troubleshooting

1. **Troubles with high memory usage**

You can try to decrease the number of CPUs in use, reducing the level of parallelism,
test it with `--num-cpus 1` flag and then increase according to your hardware.

## How to Contribute

Just open your [issues](https://github.com/icaropires/pdf2dataset/issues) and/or [pull requests](https://github.com/icaropires/pdf2dataset/pulls), all are welcome :smiley:!
