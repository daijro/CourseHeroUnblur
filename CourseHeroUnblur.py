import sys

# default values
IMAGE_UPSCALING  = False
USE_OCR          = True
OPEN_FILE        = False
PDF_FILE_NAME    = None
url              = None

# argument parser
if len(sys.argv) > 1:
    import argparse
    parser = argparse.ArgumentParser(description='Bypass the coursehero paywall')
    parser.add_argument('-l', '--url', help='The coursehero url to bypass', required=True)
    parser.add_argument('-o', '--output', help='Output file (default file name from CourseHero)', required=False)
    parser.add_argument('-n', '--no-ocr', help='Don\'t scan PDF with OCR, won\'t break without GhostScript and Tesseract', default=False, action='store_true')
    parser.add_argument('-s', '--sharpen', help='Sharpens the image for better OCR output (makes images grayscale)', default=False, action='store_true')
    parser.add_argument('--open', help='Opens the PDF in the default web browser', default=False, action='store_true')
    args = parser.parse_args()
    
    url = args.url
    IMAGE_UPSCALING = args.sharpen
    OPEN_FILE = args.open
    USE_OCR = not args.no_ocr
    PDF_FILE_NAME = args.output
    
# other imports
import grequests
from skimage.metrics import structural_similarity
import numpy as np
import re
from fake_headers import Headers
from requests_html import HTMLSession
import cv2
from pathlib import Path
from skimage.io._plugins.pil_plugin import ndarray_to_pil
from threading import Thread
from bs4 import BeautifulSoup as bs
from eta import ETA

from borb.pdf.canvas.layout.image.image import Image as Image
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF

# optional imports
if USE_OCR:
    import ocrmypdf
if OPEN_FILE:
    import webbrowser

from blessed import Terminal
from colorama import Fore, init
Terminal()
init()


log_indic = f"{Fore.RESET}[{Fore.CYAN}>{Fore.RESET}]"
err_indic = f"{Fore.RESET}[{Fore.RED}>{Fore.RESET}]"

# prompt for url if not given
if not url:
    try:
        url = input(f"{log_indic} Enter the coursehero url: ").strip()
    except KeyboardInterrupt:
        print(f"{err_indic} Exiting...")
        exit()

# input validation
if 'coursehero.com/file' not in url.lower():
    print(f"{err_indic} {Fore.RED}Invalid url. Exiting...{Fore.RESET}")
    exit()


make_headers = lambda: {key:value for key, value in Headers(headers=True).generate().items() if key != 'Accept-Encoding'}

session = HTMLSession()

try:
    resp = session.get(url, headers=make_headers())
except:
    print(f"{err_indic} {Fore.RED}Request connection failed. Exiting...{Fore.RESET}")
    exit()


try:
    pot_url = re.findall('url\\(\\/doc-asset\\/bg[\\/a-z0-9\\.\\-]+\\);', resp.text)[0][4:-2]
except IndexError:
    print(Fore.RED+bs(resp.text, features='lxml').get_text()+Fore.RESET)
    exit()
except:
    print(f"{err_indic} {Fore.RED}Failed to parse request content. Exiting...{Fore.RESET}")
    exit()

soup = bs(resp.content, features='lxml')

path = Path(PDF_FILE_NAME.rstrip('/').rstrip('\\')) if PDF_FILE_NAME else None
if PDF_FILE_NAME and not path.is_dir():
    # Check if given path is a directory
    if path.is_file(): # if file
        PDF_FILE_NAME = str(path)
    elif not path.exists(): # if path doesn't exist yet
        PDF_FILE_NAME = (Path().cwd() / PDF_FILE_NAME).absolute()
else:
    sluggify = lambda s: '_'.join(re.sub(r'[^\w\d\s]+', '', s).split()) + '.pdf'
    
    dir = path or Path().cwd()
    PDF_FILE_NAME = dir / sluggify(soup.find('h1', class_='bdp_title_heading').text)
        
    while PDF_FILE_NAME.exists():
        PDF_FILE_NAME = Path(f'{str(PDF_FILE_NAME)[:-4]}-1{str(PDF_FILE_NAME)[-4:]}')

    PDF_FILE_NAME = str(PDF_FILE_NAME)
                

endLink = (pot_url
    .replace("background-image:", "")
    .replace("-html-bg", "")
    .replace(" ", "")
)
endLink = '/'.join(endLink.split('/')[:-1])+'/'
dataRSID = re.findall('.*\\/', pot_url[pot_url.find('splits/')+7:])[0][:-1]
numberDataRsid = "v9" not in endLink


try:
    pageAmount = int(soup.find('label', text='Pages').parent.text.split()[-1])
except AttributeError:
    print(f"{err_indic} Couldn't successfully get the page amount. Guessing 2 by default.\n")
    pageAmount = 2

print(f"{log_indic} File's Info")
print(f"{log_indic} Uses numberDataRsid: {numberDataRsid}")
print(f"    {log_indic} Data-RSID: {dataRSID}\n    {log_indic} Page-Amount: {pageAmount}\n    {log_indic} Path: {endLink}")


# generating possible links
generatedPageList = []
fullBlurredPages = []
validLinks = 0

if numberDataRsid:
    npurls = [f"https://www.coursehero.com{endLink}".replace(dataRSID, str(int(dataRSID)+n)) for n in [0, 1, -1]]
else:
    npurls = []

purls = [
    f"https://www.coursehero.com{endLink}".replace(dataRSID, "v9"),
    f"https://www.coursehero.com{endLink}".replace(dataRSID, "v9.2"),
]


# get working blurred pages
print(f"\n{log_indic} {Fore.CYAN}Gathering {pageAmount} potential full blurred page{'s' if pageAmount != 1 else ''}...{Fore.RESET}")


def test_sites(url_list):
    return [
            url.url
            for url in grequests.map(
                [
                    grequests.head(url, headers=make_headers())
                    for url in url_list
                ],
                size=len(url_list),
            )
            if url and url.status_code == 200
        ]

print(Fore.LIGHTBLACK_EX, end='')
eta = ETA(pageAmount)

def getBlurredPages(page):
    global generatedPageList, fullBlurredPages

    blurredPages = []
    split_range = range(page+3)
    for blurredPage in [
        [f"{url}split-{n}-page-{page}.jpg" for n in split_range] for url in purls
    ] + [
        # If the given file does have an actual data rsid
        [f"{url}split-{n}-page-{page}.jpg" for n in split_range] for url in npurls
    ]:
        if blurredPage := test_sites(blurredPage):
            blurredPages.extend(blurredPage)
            if blurredPage:
                break        
    
    if blurredPages:
        for fullBlurredPage in [[[f"{url}page-{page}.jpg"] for url in urls] for urls in [purls, npurls]]:
            if fullBlurredPage := test_sites(fullBlurredPage):
                fullBlurredPages.append(fullBlurredPage[0])
                break
        else:
            fullBlurredPages.append(f'{blurredPages[-1][:-4]}-html-bg{blurredPages[-1][-4:]}')
            eta.print_status(extra=f'Finished page #{page}\t')
        generatedPageList.extend(blurredPages)

full_page_threads = []
for page in range(1, pageAmount+1):
    full_page_threads.append(Thread(target=getBlurredPages, args=(page,)))
    full_page_threads[-1].daemon = True
    full_page_threads[-1].start()

for t in full_page_threads:
    t.join()

eta.done()
print(Fore.RESET, end='')

def pg_num_from_link(url):
    return int(re.search('page-\d+', url).group(0)[5:])

def split_from_link(url):
    return int(re.search('split-\d+', url).group(0)[6:])


generatedPageList = sorted(generatedPageList, key=lambda x: (pg_num_from_link(x), split_from_link(x)))
fullBlurredPages = sorted(fullBlurredPages, key=lambda x: pg_num_from_link(x))


print(f"\n{log_indic} {Fore.GREEN}{len(fullBlurredPages)} page{'s' if len(fullBlurredPages) != 1 else ''} found.{Fore.RESET}")
for p in fullBlurredPages: print(f'{Fore.LIGHTBLACK_EX}* {p}{Fore.RESET}')

print(f"\n{log_indic} {Fore.GREEN}{len(generatedPageList)} page split{'s' if len(generatedPageList) != 1 else ''} found.{Fore.RESET}")
for p in generatedPageList: print(f'{Fore.LIGHTBLACK_EX}* {p}{Fore.RESET}')


if not fullBlurredPages or not generatedPageList:
    print(f'{err_indic} {Fore.RED}Exiting...{Fore.RESET}')
    exit()

print(f"\n{log_indic} {Fore.CYAN}Building pages...{Fore.RESET}")


def upscale_image(img):
    if not IMAGE_UPSCALING:
        return img
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    return img


print(Fore.LIGHTBLACK_EX, end='')
eta = ETA(len(fullBlurredPages))


def stitch_page(n, page_url):
    p = pg_num_from_link(page_url)
    valid_pages = [url for url in generatedPageList if pg_num_from_link(url) == p]
    # get content as pillow image
    im_parts = [
        upscale_image(cv2.imdecode(np.asarray(bytearray(resp.content), dtype=np.uint8), -1))
        for resp in grequests.map(
            [grequests.get(url, headers=make_headers()) for url in valid_pages+[page_url]],
            size=5
        )
        if resp and resp.status_code == 200
    ]
    after_parts, before = im_parts[:-1], im_parts[-1]
    new_img = ndarray_to_pil(before).convert('RGB')
    
    if not IMAGE_UPSCALING:
        before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    
    for after in im_parts:
        # Convert images to grayscale and compute SSIM between two images
        diff = structural_similarity(
            before,
            # image will already be grayscale when upscaling is applied
            after if IMAGE_UPSCALING else cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
            full=True
        )[1]
        
        diff = (diff * 255).astype("uint8")
        
        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]        
        
        mask = np.zeros(before.shape, dtype='uint8')
        
        for c in contours:
            if cv2.contourArea(c) > 40:
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                # rectangle boundary selections:
                # x, y, w, h = cv2.boundingRect(c)
                # cv2.rectangle(mask, (x, y), (x + w, y + h), (255,255,255), -1)
                
        new_img.paste(ndarray_to_pil(after).convert('RGB'), (0, 0), ndarray_to_pil(mask).convert('L'))
    
    imgs[n] = new_img
    eta.print_status(extra=f"Finished page #{n}\t")


threads = []
imgs = {}

for n, page_url in enumerate(fullBlurredPages):
    threads.append(Thread(target=stitch_page, args=(n, page_url)))
    threads[-1].daemon = True
    threads[-1].start()

for t in threads:
    t.join()
    # print(f"{Fore.LIGHTBLACK_EX}Finished stitching page #{n+1}{Fore.RESET}")
eta.done()
print(Fore.RESET, end='')


imgs = [i[1] for i in sorted(imgs.items())]

print(f"\n{log_indic} {Fore.CYAN}Converting to PDF...{Fore.RESET}")

# Create PDF
doc = Document()

for img in imgs:
    # Create/add Page
    page = Page(img.width + 10, img.height + 10)
    doc.append_page(page)

    # Set PageLayout
    layout = SingleColumnLayout(page, horizontal_margin=0, vertical_margin=0)

    # Add Image
    layout.add(Image(img))


# write to disk
print(f"\n{log_indic} {Fore.CYAN}Saving PDF as {Fore.BLUE}{PDF_FILE_NAME}{Fore.RESET}")
with open(PDF_FILE_NAME, "wb") as pdf_file_handle:
    PDF.dumps(pdf_file_handle, doc)
        
if USE_OCR:
    try:
        print(f"\n{log_indic} {Fore.CYAN}Running OCR...{Fore.RESET}")
        ocrmypdf.ocr(PDF_FILE_NAME, PDF_FILE_NAME, use_threads=True)
    except FileNotFoundError:
        print(f"{err_indic} {Fore.RED}OCR failed. Please make sure Ghostscript and Tesseract-OCR are installed.{Fore.RESET}")

if OPEN_FILE:
    print(f"\n{log_indic} {Fore.CYAN}Opening in web browser...{Fore.RESET}")
    webbrowser.open(PDF_FILE_NAME)


print(f"\n{Fore.GREEN}Complete.{Fore.RESET}")