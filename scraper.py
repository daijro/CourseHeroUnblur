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
import gevent
from bs4 import BeautifulSoup as bs
from eta import ETA
import datetime
import time
import subprocess


def _make_headers():
    """
    Returns a dictionary of headers to be used in requests
    """
    h = dict(Headers(headers=True).generate().items())
    h['Accept-Encoding'] = 'deflate'
    return h


class ETA_(ETA):
    # Modify ETA class to be callable from a GUI
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step   = 0  # do not incrememnt on self.print_status()
        self.done_  = False

    message = 'Running...'
    def tick(self, msg='', step=1):
        self.message    = msg
        self.last_step  += step
    
    def print_status(self):
        return super().print_status(extra=self.message)
    
    def print_status_loop(self):
        while not self.done_:
            self.print_status()
            time.sleep(self.min_ms_between_updates / 1000)
            
    def _thread_eta(self):
        self.thread = Thread(target=self.print_status_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def done(self):
        self.done_ = True
        return super().done()
    
    def get_elapsed(self):
        now = datetime.datetime.now()
        if not self._started:
            self._started = now
            return 0
        else:
            td = now - self.started
            return (td.days * 86400) + td.seconds
        
    def get_remaining(self):            
        return (
            self.ave_remaining(self.last_step, elapsed_sec)
            if (elapsed_sec := self.get_elapsed())
            else None
        )


class PHASE1:
    """
    CHU PHASE 1:
    GATHER DOCUMENT DETAILS
    """
    class exceptions:
        InvalidURL      = Exception("Invalid URL")
        RequestsError   = Exception("Requests Error; Couldn't get CourseHero page information")
        TooManyRequests = Exception("Too many requests have been sent. Please rotate your IP or try again later")
        FailedToParse   = Exception("Failed to parse CourseHero page information")
        PageAmountError = Exception("Could not find the document page amount")
        InvalidPDFName  = Exception("File name must end in .pdf")
        
        
    def __init__(self, URL, PDF_FILE_NAME=None):
        self.url            = URL
        self.pdf_file_name  = PDF_FILE_NAME


    def run(self):
        self.input_validation()
        self.request_page()
        self.set_initial_information()
        self.set_pdf_file_name()
        
        del self._session, self._resp, self._soup
        
        """
        VARIABLES:
        self.numberDataRsid
        self.dataRSID
        self.pageAmount
        self.linkPath
        self.pdf_file_name
        """        


    def input_validation(self):
        """
        Checks if the given URL is valid
        """
        if 'coursehero.com/file' not in self.url.lower():
            raise self.exceptions.InvalidURL

        
    def request_page(self):
        """
        self._session, self._resp, self._soup
        """
        self._session = HTMLSession()
        try:
            self._resp = self._session.get(self.url, headers=_make_headers())
        except Exception as e:
            raise self.exceptions.RequestsError from e
        self._soup = bs(self._resp.content, features='lxml')
    
    
    def set_initial_information(self):
        """
        self.numberDataRsid, self.dataRSID, self.linkPath
        """
        try:
            linkPath = pot_url = re.findall(r'\/doc-asset\/bg\/[a-z0-9\.\-]+\/splits\/\d+\/split\-[a-z0-9\.\-]+\.jpg', self._resp.text)[0]
        except IndexError as e:
            raise self.exceptions.TooManyRequests from e
        except Exception as e:
            raise self.exceptions.FailedToParse from e
        
        try:
            self.pageAmount = int(self._soup.find('label', text='Pages').parent.text.split()[-1])
        except AttributeError as e:
            raise self.exceptions.PageAmountError from e
        
        # remove all the unnecessary stuff
        for s in [
            "background-image:",
            "-html-bg",
            " ",
        ]:
            linkPath = re.sub(s, '', linkPath)
        self.linkPath = '/'.join(linkPath.split('/')[:-1])+'/'

        self.dataRSID = re.findall('.*\\/', pot_url[pot_url.find('splits/')+7:])[0][:-1]
        self.numberDataRsid = "v9" not in linkPath
        

    def set_pdf_file_name(self):
        """
        self.pdf_file_name
        """
        path = Path(self.pdf_file_name.rstrip('/').rstrip('\\')) if self.pdf_file_name else None
        if self.pdf_file_name and not path.is_dir():
            if not str(self.pdf_file_name).lower().endswith('.pdf'):
                raise self.exceptions.InvalidPDFName
            if path.is_file(): # if file
                self.pdf_file_name = str(path)
            elif not path.exists(): # if path doesn't exist yet
                self.pdf_file_name = (Path().cwd() / self.pdf_file_name).absolute()
        else:
            # If given path is a directory
            dir = path or Path().cwd()
            self.pdf_file_name = dir / '_'.join(re.sub(r'[^\w\d\s]+', '', 
                self._soup.find('h1', class_='bdp_title_heading').text
            ).split())
                
            if Path(f'{self.pdf_file_name}.pdf').exists():
                n = -1
                while Path(f'{self.pdf_file_name}{n}.pdf').exists():
                    n -= 1
                self.pdf_file_name = f'{self.pdf_file_name}{n}'
                    
            self.pdf_file_name = f'{self.pdf_file_name}.pdf'


class PHASE2:
    """
    CHU PHASE 2:
    - FIND WORKING PAGES
    - STITCH PAGES
    - CONVERT TO PIL IMAGE
    """
    class exceptions:
        NoPagesFound     = Exception("No pages found")
        NoSplitsFound    = Exception("Couldn't find splits for this page")
        InvalidPageRange = Exception("Given page range is invalid")
        PageOutOfRange   = Exception("Given page range is out of range")
        NoPagesFound     = Exception("Failed to find any pages")
        DuplicatePages   = Exception("Given page range has dubplicate pages")
    
    imgs = {}
    
    def __init__(self, numberDataRsid, dataRSID, linkPath, pageAmount, IMAGE_UPSCALING, PAGE_RANGE=None, DEBUG=False):
        self.pageAmount:      int  = pageAmount
        self.IMAGE_UPSCALING: bool = IMAGE_UPSCALING
        self.debug:           bool = DEBUG
        self.PAGE_RANGE            = self.parse_page_range(PAGE_RANGE) if PAGE_RANGE else range(1, self.pageAmount+1)

        # generating possible links
        if numberDataRsid:
            self.npurls = [f"https://www.coursehero.com{linkPath}".replace(dataRSID, str(int(dataRSID)+n)) for n in [0, 1, -1]]
        else:
            self.npurls = []

        self.purls = [
            f"https://www.coursehero.com{linkPath}".replace(dataRSID, "v9"),
            f"https://www.coursehero.com{linkPath}".replace(dataRSID, "v9.2"),
        ]


    def parse_page_range(self, s):
        range_num = []
        for r in s.rstrip(',').split(','):
            try:
                r = [int(i) for i in r.split('-')]
            except ValueError as e:
                raise self.exceptions.InvalidPageRange from e
            # error checking
            if len(r) not in range(1, 3) or min(r) < 1:
                raise self.exceptions.InvalidPageRange
            elif max(r) > self.pageAmount:
                raise self.exceptions.PageOutOfRange
            n = -1 if r[0] > r[-1] else 1
            range_num.extend(range(r[0], r[-1]+n, n))
        # finding duplicates
        if len(set(range_num)) != len(range_num):
            raise self.exceptions.DuplicatePages
        return range_num
        

    def run(self, print_eta=False):
        self.eta = ETA_(self.pageAmount*2)
        if print_eta:
            self.eta._thread_eta()
        group = gevent.pool.Pool(5)
        for page in sorted(self.PAGE_RANGE):
            group.add(gevent.spawn(self.set_imgs, page))
        group.join()
        
        self.eta.done()
        
        if not self.imgs:
            raise self.exceptions.NoPagesFound
        
        self.imgs = [i[1] for i in sorted(
                self.imgs.items(), key=lambda x: self.PAGE_RANGE.index(x[0])
            )]


    def test_sites(self, url_list):
        return [
                url.url
                for url in grequests.map(
                    [
                        grequests.head(url, headers=_make_headers())
                        for url in url_list
                    ],
                    size=len(url_list),
                )
                if url and url.status_code == 200
            ]


    def set_imgs(self, page):
        blurredPages = []
        split_range = range(page+3)
        for blurredPage in [
            [f"{url}split-{n}-page-{page}.jpg" for n in split_range] for url in self.purls
        ] + [
            # If the given file does have an actual data rsid
            [f"{url}split-{n}-page-{page}.jpg" for n in split_range] for url in self.npurls
        ]:
            if blurredPage := self.test_sites(blurredPage):
                blurredPages.extend(blurredPage)
                if blurredPage:
                    break
        
        if not blurredPages:
            self.eta.tick(msg=f'Failed to stitch page #{page}', step=2)
            return
        
        blurredPages = sorted(blurredPages, key=lambda x: self._split_from_link(x))
        for fullBlurredPage in [[[f"{url}page-{page}.jpg"] for url in urls] for urls in [self.purls, self.npurls]]:
            if fullBlurredPage := self.test_sites(fullBlurredPage):
                fullpageURL = fullBlurredPage[0]
                break
        else:
            fullpageURL = f'{blurredPages[-1][:-4]}-html-bg{blurredPages[-1][-4:]}'
        self.eta.tick(msg=f'Found {len(blurredPages)+1} pieces for page #{page}')
        
        try:
            self.imgs[page] = self.stitch_page(fullpageURL, blurredPages)
        except Exception as e:
            self.eta.tick(msg=f'Failed to get splits for page #{page}')
            if self.debug:
                raise self.exceptions.NoSplitsFound from e
        else:
            self.eta.tick(msg=f'Page #{page} complete')
            

    def _pg_num_from_link(self, url):
        return int(re.search('page-\d+', url).group(0)[5:])


    def _split_from_link(self, url):
        return int(re.search('split-\d+', url).group(0)[6:])


    def get_upscale_image(self, img):
        # skip image upscaling if not enabled
        if not self.IMAGE_UPSCALING:
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


    def stitch_page(self, page_url, valid_pages):
        p = self._pg_num_from_link(page_url)
        # get content as pillow image
        im_parts = [
            self.get_upscale_image(cv2.imdecode(np.asarray(bytearray(resp.content), dtype=np.uint8), -1))
            for resp in grequests.map(
                [grequests.get(url, headers=_make_headers()) for url in valid_pages+[page_url]],
                size=5,
            )
            if resp and resp.status_code == 200
        ]
        after_parts, before = im_parts[:-1], im_parts[-1]
        new_img = ndarray_to_pil(before).convert('RGB')
        
        if not self.IMAGE_UPSCALING:
            before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        
        for after in im_parts:
            # Convert images to grayscale and compute SSIM between two images
            diff = structural_similarity(
                before,
                # image will already be grayscale when upscaling is applied
                after if self.IMAGE_UPSCALING else cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
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
        
        return new_img


class PHASE3:
    """
    CHU PHASE 3:
    - WRITE IMAGES TO PDF DOCUMENT
    - SAVE FILE AND RUN OCR AS NEEDED
    """
    class exceptions:
        OCRFailed = Exception('OCR failed. Please make sure OCRmyPDF is installed in PATH, along with its dependencies Ghostscript & Tesseract-OCR')
        
    def __init__(self, pdf_file_name, imgs, USE_OCR):
        self.pdf_file_name  = pdf_file_name
        self.use_ocr        = USE_OCR
        self.imgs           = imgs

    def run(self, debug=False):
        # write to disk
        Path(self.pdf_file_name).parent.mkdir(parents=True, exist_ok=True)

        self.imgs[0].save(self.pdf_file_name, "PDF", resolution=100.0, save_all=True, append_images=self.imgs[1:])
                
        if self.use_ocr:
            try:
                cmd_args = ['ocrmypdf'] + (
                    [] if debug else ['-q']
                    ) + [self.pdf_file_name, self.pdf_file_name]
                subprocess.run(cmd_args)
            except FileNotFoundError as e:
                raise self.exceptions.OCRFailed from e
