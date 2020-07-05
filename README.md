# pdf2dataset

![pdf2dataset](https://github.com/icaropires/pdf2dataset/workflows/pdf2dataset/badge.svg)

For extracting text from PDFs and save to a dataset

## Install

### Install Dependencies

#### Ubuntu (or debians)

``` bash
$ sudo apt update
$ sudo apt install -y poppler-utils tesseract-ocr-por
```

## Usage examples

``` bash
# Reads all PDFs from my_pdfs_folder and saves the resultant dataframe to my_df.parquet.gzip
$ pdf2dataset my_pdfs_folder my_df.parquet.gzip
```

## Install

### Usage

``` bash
# first, clone repository

$ pip install . --user  # Please, isolate the environment
```


### Development

``` bash
# first, clone repository

$ poetry install
```
