# CourseHeroUnblur

Download unblurred CourseHero documents as searchable PDFs

<hr width=50>


## How it works

This tool gathers all the avaliable previews of a document in CourseHero, then uses image manipulation to identify and switch together the unblurred parts of each page to recreate the file behind the paywall. These images are then (optionally) upscaled and saved to a OCR-scanned searchable PDF file.

<hr width=50>

## In Action

![in action](https://i.imgur.com/ekBma2W.mp4)


<hr width=50>

## Installation

**Tested on [Python 3.8.9](https://www.python.org/downloads/release/python-389/)**

Pip packages:
```
html5lib
bs4
requests-html
fake-headers
grequests
pillow
opencv-python
numpy
colorama
blessed
eta
scikit-image
borb
ocrmypdf
```

*Optional for selectable text in PDF:*

- Ghostscript: ([downloads](https://www.ghostscript.com/releases/gsdnld.html))
- Tesseract-OCR (add to PATH): ([downloads](https://tesseract-ocr.github.io/tessdoc/Home.html#binaries))

---

## Usage


### CLI

Command line arguments:
```
usage: CourseHeroUnblur.py [-h] -l URL [-o OUTPUT] [-n] [-s] [--open]

Bypass the coursehero paywall

optional arguments:
  -h, --help            show this help message and exit
  -l URL, --url URL     The coursehero url to bypass
  -o OUTPUT, --output OUTPUT
                        Output file (default file name from CourseHero)
  -n, --no-ocr          Don't scan PDF with OCR, won't break without GhostScript and Tesseract
  -s, --sharpen         Sharpens images for better OCR output (makes images grayscale)
  --open                Opens the PDF in the default web browser
```

---


### Disclaimer
The purpose of this program is to provide an example of asynchronous webscraping and data gathering in Python. I am not responsible for any misuse of this tool. This tool was created strictly for educational purposes only.