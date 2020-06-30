# Extract text

![pdf2dataset](https://github.com/icaropires/pdf2dataset/workflows/pdf2dataset/badge.svg)

For extracting text from PDFs and save to a dataset

## Install

### Install Dependencies

#### Ubuntu (or debians)

``` bash
$ sudo apt update
$ sudo apt install -y poppler-utils tesseract-ocr-por
$ pip3 install -r requirements.txt --user  # Please isolate the environment
```

## Usage example

``` bash
python3 pdf2dataset/extract.py pdfs txts
```
