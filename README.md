# pdf2dataset

[![pdf2dataset](https://github.com/icaropires/pdf2dataset/workflows/pdf2dataset/badge.svg?branch=master)](https://github.com/icaropires/pdf2dataset)
[![pypi](https://img.shields.io/pypi/v/pdf2dataset.svg)](https://pypi.python.org/pypi/pdf2dataset)
[![Maintainability](https://api.codeclimate.com/v1/badges/cbe90c3043b038f52b18/maintainability)](https://codeclimate.com/github/icaropires/pdf2dataset/maintainability)
[![codecov](https://codecov.io/gh/icaropires/pdf2dataset/branch/master/graph/badge.svg)](https://codecov.io/gh/icaropires/pdf2dataset)
[![pypi-stats](https://img.shields.io/pypi/dm/pdf2dataset)](https://pypistats.org/packages/pdf2dataset)

Converts a whole subdirectory with any volume (small or huge) of PDF documents to a dataset (pandas DataFrame).
No need to setup any external service (no database, brokers, etc). Just install and run it!


## Main features

* Conversion of a whole subdirectory with PDFs documents into a pandas DataFrame
* Support for parallel and distributed processing through [ray](https://github.com/ray-project/ray)
* Extractions are performed by page, making tasks distribution more uniform for handling documents with big differences in number of pages
* Incremental writing of resulting DataFrame, making possible to process data bigger than memory
* Error tracking of faulty documents
* Resume interrupted processing
* Extract text through [pdftotext](https://github.com/jalan/pdftotext)
* Use OCR for extracting text through [pytesseract](https://github.com/madmaze/pytesseract)
* Extract images through [pdf2image](https://github.com/Belval/pdf2image)
* Support to implement custom features extraction
* Highly customizable behavior through params


## Installation

### Install Dependencies

#### Fedora

``` bash
# "-por" for portuguese, use the documents language
$ sudo dnf install -y gcc-c++ poppler-utils pkgconfig poppler-cpp-devel python3-devel tesseract-langpack-por
```

#### Ubuntu (or debians)

``` bash
$ sudo apt update

# "-por" for portuguese, use the documents language
$ sudo apt install -y build-essential poppler-utils libpoppler-cpp-dev pkg-config python3-dev tesseract-ocr-por
```

### Install pdf2dataset

#### For usage

``` bash
$ pip3 install pdf2dataset --user  # Please, isolate the environment
```

#### For development

``` bash
# First, install poetry, clone repository and cd into it
$ poetry install
```


## Usage

### Simple - CLI

``` bash
# Note: path, page and error will always be present in resulting DataFrame

# Reads all PDFs from my_pdfs_dir and saves the resultant dataframe to my_df.parquet.gzip
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip  # Most basic, extract all possible features
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip --features=text  # Extract just text
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip --features=image  # Extract just image
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip --num-cpus 1  # Maximum reducing of parallelism
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip --ocr true  # For scanned PDFs
$ pdf2dataset my_pdfs_dir my_df.parquet.gzip --ocr true --lang eng  # For scanned documents with english text
```

### Resume processing

In case of any interruption, to resume the processing, just use the same path as output and the
processing will be resumed automatically. The flag `--saving-interval` (or the param `saving_interval`)
controls the frequency the output path will be updated, and so, the processing "checkpoints".


### Using as a library

#### Main functions

There're some helper functions to facilitate pdf2dataset usage:

* **extract:** function can be used analogously to the CLI
* **extract_text**: `extract` wrapper with `features=text`
* **extract_image**: `extract` wrapper with `features=image`
* **image_from_bytes:** (pdf2image.utils) get a Pillow `Image` object given the image bytes
* **image_to_bytes:** (pdf2image.utils) get the image bytes given the a Pillow `Image` object

#### Basic example
``` python
from pdf2dataset import extract

extract('my_pdfs_dir', 'all_features.parquet.gzip')
```

#### Small data

One feature, not available to the CLI, is the custom behavior for handling small volumes of data (small can
be understood as that: the extraction won't run for hours or days and won't be distributed).

The complete list of differences are:

* Faster initialization (use multiprocessing instead of ray)
* Don't save processing progress
* Distributed processing not supported
* Don't write dataframe to disk
* Returns the dataframe

##### Example:
``` python
from pdf2dataset import extract_text

df = extract_text('my_pdfs_dir', small=True)
# ...
```

#### Pass list of files paths

Instead of specifying a directory, one can specify a list of files to be processed.

##### Example:

``` python
from pdf2dataset import extract


my_files = [
    './tests/samples/single_page1.pdf',
    './tests/samples/invalid1.pdf',
]

df = extract(my_files, small=True)
# ...
```

#### Pass files from memory

If you don't want to specify a directory for the documents, you can specify the tasks that
will be processed.

The tasks can be of the form `(document_name, document_bytes, page_number)`
or just `(document_name, document_bytes)`, **document_name** must ends with `.pdf` but 
don't need to be a real file, **document_bytes** are the bytes of the pdf document and
**page_number** is the number of the page to process (all pages, if not specified).

##### Example:

``` python
from pdf2dataset import extract_text

tasks = [
    ('a.pdf', a_bytes),  # Processing all pages of this document
    ('b.pdf', b_bytes, 1),
    ('b.pdf', b_bytes, 2),
]

# 'df' will contain results from all pages from 'a.pdf' and page 1 and 2 from 'b.pdf'
df = extract_text(tasks, 'my_df.parquet.gzip', small=True)

# ...
```

#### Returning a list

If you don't want to handle the DataFrame, is possible to return a nested list with the features values.
The structure for the resulting list is:
```
result = List[documents]
documents = List[pages]
pages = List[features]
features = List[feature]
feature = any
```

* `any` is any type supported by pyarrow.
* features are ordered by the feature name (`text`, `image`, etc)

##### Example:

``` python
>>> from pdf2dataset import extract_text
>>> extract_text('tests/samples', return_list=True)
[[[None]],
 [['First page'], ['Second page'], ['Third page']],
 [['My beautiful sample!']],
 [['First page'], ['Second page'], ['Third page']],
 [['My beautiful sample!']]]
```

* Features with error will have `None` value as result
* Here, `extract_text` was used, so the only feature is `text`

#### Custom Features

With version >= 0.4.0, is also possible to easily implement extraction of custom features:

##### Example:

This is the structure:

``` python
from pdf2dataset import extract, feature, PdfExtractTask


class MyCustomTask(PdfExtractTask):

    @feature('bool_')
    def get_is_page_even(self):
        return self.page % 2 == 0

    @feature('binary')
    def get_doc_first_bytes(self):
        return self.file_bin[:10]

    @feature('string', exceptions=[ValueError])
    def get_wrong(self):
        raise ValueError("There was a problem!")


if __name__ == '__main__':
    df = extract('tests/samples', small=True, task_class=MyCustomTask)
    print(df)

    df.dropna(subset=['text'], inplace=True)  # Discard invalid documents
    print(df.iloc[0].error)
```

* First print:
```
                         path  page doc_first_bytes  ...                  text  wrong                                              error
0                invalid1.pdf    -1   b"I'm invali"  ...                  None   None  image_original:\nTraceback (most recent call l...
1             multi_page1.pdf     2  b'%PDF-1.5\n%'  ...           Second page   None  wrong:\nTraceback (most recent call last):\n  ...
2             multi_page1.pdf     3  b'%PDF-1.5\n%'  ...            Third page   None  wrong:\nTraceback (most recent call last):\n  ...
3   sub1/copy_multi_page1.pdf     1  b'%PDF-1.5\n%'  ...            First page   None  wrong:\nTraceback (most recent call last):\n  ...
4   sub1/copy_multi_page1.pdf     3  b'%PDF-1.5\n%'  ...            Third page   None  wrong:\nTraceback (most recent call last):\n  ...
5             multi_page1.pdf     1  b'%PDF-1.5\n%'  ...            First page   None  wrong:\nTraceback (most recent call last):\n  ...
6  sub2/copy_single_page1.pdf     1  b'%PDF-1.5\n%'  ...  My beautiful sample!   None  wrong:\nTraceback (most recent call last):\n  ...
7   sub1/copy_multi_page1.pdf     2  b'%PDF-1.5\n%'  ...           Second page   None  wrong:\nTraceback (most recent call last):\n  ...
8            single_page1.pdf     1  b'%PDF-1.5\n%'  ...  My beautiful sample!   None  wrong:\nTraceback (most recent call last):\n  ...

[9 rows x 8 columns]
```

* Second print:
```
wrong:
Traceback (most recent call last):
  File "/home/icaro/Desktop/pdf2dataset/pdf2dataset/extract_task.py", line 32, in inner
    result = feature_method(*args, **kwargs)
  File "example.py", line 16, in get_wrong
    raise ValueError("There was a problem!")
ValueError: There was a problem!

```

Notes:
* `@feature` is the decorator used to define new features.
* The extraction method name must start with the prefix `get_` (avoids collisions with attribute names and increases readability)
* First argument to `@feature` must be a valid PyArrow type, complete list [here](https://arrow.apache.org/docs/python/api/datatypes.html)
* `exceptions` param specify a list of exceptions to be recorded on DataFrame, otherwise they are raised
* For this example, all available features plus the custom ones are extracted


### Results File

The resulting "file" is a directory with structure specified by dask with pyarrow engine,
it can be easily read with pandas or dask:

#### Example with pandas
``` python
>>> import pandas as pd
>>> df = pd.read_parquet('my_df.parquet.gzip', engine='pyarrow')
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

There is no guarantee about the uniqueness or order of `index`, you might need to create a new index with
the whole data in memory.

The `-1` page number means that was not possible of even parsing the document.

### Run on a Cluster

#### Setup the Cluster

Follow ray documentation for [manual](https://docs.ray.io/en/latest/using-ray-on-a-cluster.html?setup#manual-cluster-setup) or [automatic](https://docs.ray.io/en/latest/autoscaling.html?setup#automatic-cluster-setup)
setup.

#### Run it

To go distributed you can run it just like local, but using the `--address` and `--redis-password` flags to point to your cluster ([More information](https://docs.ray.io/en/latest/multiprocessing.html)).

With version >= 0.2.0, only the head node needs to have access to the documents in disk.


### CLI Help

```
usage: pdf2dataset [-h] [--features FEATURES]
                   [--saving-interval SAVING_INTERVAL] [--ocr-lang OCR_LANG]
                   [--ocr OCR] [--chunksize CHUNKSIZE]
                   [--image-size IMAGE_SIZE] [--ocr-image-size OCR_IMAGE_SIZE]
                   [--image-format IMAGE_FORMAT] [--num-cpus NUM_CPUS]
                   [--address ADDRESS] [--dashboard-host DASHBOARD_HOST]
                   [--redis-password REDIS_PASSWORD]
                   input_dir out_file

Extract text from all PDF files in a directory

positional arguments:
  input_dir             The folder to lookup for PDF files recursively
  out_file              File to save the resultant dataframe

optional arguments:
  -h, --help            show this help message and exit
  --features FEATURES   Specify a comma separated list with the features you
                        want to extract. 'path' and 'page' will always be
                        added. Available features to add: image, page, path,
                        text Examples: '--features=text,image' or '--
                        features=all'
  --saving-interval SAVING_INTERVAL
                        Results will be persisted to results folder every
                        saving interval of pages
  --ocr-lang OCR_LANG   Tesseract language
  --ocr OCR             'pytesseract' if true, else 'pdftotext'. default:
                        false
  --chunksize CHUNKSIZE
                        Chunksize to use while processing pages, otherwise is
                        calculated
  --image-size IMAGE_SIZE
                        If adding image feature, image will be resized to this
                        size. Provide two integers separated by 'x'. Example:
                        --image-size 1000x1414
  --ocr-image-size OCR_IMAGE_SIZE
                        The height of the image OCR will be applied. Width
                        will be adjusted to keep the ratio.
  --image-format IMAGE_FORMAT
                        Format of the image generated from the PDF pages
  --num-cpus NUM_CPUS   Number of cpus to use
  --address ADDRESS     Ray address to connect
  --dashboard-host DASHBOARD_HOST
                        Which IP ray webui will try to listen on
  --redis-password REDIS_PASSWORD
                        Redis password to use to connect with ray
```


## Troubleshooting

1. **Troubles with high memory usage**

* Decrease the number of CPUs in use, reducing the level of parallelism, test it with `--num-cpus 1` flag and then increase according to your hardware.

* Use smaller chunksize, so less documents will be put in memory at once. Use `--chunksize 1` for having `1 * num_cpus` documents in memory at once.


## How to Contribute

Just open your [issues](https://github.com/icaropires/pdf2dataset/issues) and/or [pull requests](https://github.com/icaropires/pdf2dataset/pulls), all are welcome :smiley:!
