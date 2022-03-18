# CourseHeroUnblur

Download unblurred CourseHero documents as searchable PDFs

<hr width=50>


## How it works

This tool gathers all the avaliable previews of a document in CourseHero, then uses image manipulation to identify and switch together the unblurred parts of each page to recreate the file behind the paywall. These images are then (optionally) upscaled and saved to a OCR-scanned searchable PDF file.


## In Action

https://user-images.githubusercontent.com/72637910/157621460-10ab458a-74f2-4334-9312-c63462742bdb.mp4


---

## Installation

**Tested on [Python 3.8.9](https://www.python.org/downloads/release/python-389/)**

#### Pip packages

Install using `requirements.txt`:
```
pip install -r requirements.txt
```

<hr width=50>

#### Other third-party dependencies

Optional dependenices for selectable text in PDF (*add to PATH*):

- Ghostscript: ([download](https://www.ghostscript.com/releases/gsdnld.html))
- Tesseract-OCR: ([download](https://tesseract-ocr.github.io/tessdoc/Home.html#binaries))

##### Using Chocolatey

Using [chocolatey](https://chocolatey.org/), you can install these dependencies by running the following command as admin:
```ps
choco install ghostscript tesseract-ocr -y
```


---

## Usage


#### CLI

Command line arguments:
```
usage: CourseHeroUnblur.py [-h] -l URL [-o OUTPUT] [-n] [-s] [-p PAGES] [--open] [--debug]

Bypass the coursehero paywall

optional arguments:
  -h, --help            show this help message and exit
  -l URL, --url URL     The coursehero url to bypass
  -o OUTPUT, --output OUTPUT
                        Output file (default file name from CourseHero)
  -n, --no-ocr          Don't scan PDF with OCR, won't break without GhostScript and Tesseract
  -s, --sharpen         Sharpens the image for better OCR output (makes images grayscale)
  -p PAGES, --pages PAGES
                        Specify a page range (example: 1,2,3-5)
  --open                Opens the PDF in the default web browser
  --debug               Show error traceback
```

---


### Disclaimer
The purpose of this program is to provide an example of asynchronous webscraping and data gathering in Python. I am not responsible for any misuse of this tool. This tool was created strictly for educational purposes only.
