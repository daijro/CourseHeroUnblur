import argparse
from blessed import Terminal
from colorama import Fore, Back, init
Terminal() # colorama support on cmd.exe
init()

parser = argparse.ArgumentParser(description='Bypass the coursehero paywall')
parser.add_argument('-l', '--url', help='The coursehero url to bypass', required=True)
parser.add_argument('-o', '--output', help='Output file (default file name from CourseHero)', required=False)
parser.add_argument('-n', '--no-ocr', help='Don\'t scan PDF with OCR, won\'t break without GhostScript and Tesseract', default=False, action='store_true')
parser.add_argument('-s', '--sharpen', help='Sharpens the image for better OCR output (makes images grayscale)', default=False, action='store_true')
parser.add_argument('-p', '--pages', help='Specify a page range (example: 1,2,3-5)', default=None)
parser.add_argument('--open', help='Opens the PDF in the default web browser', default=False, action='store_true')
parser.add_argument('--debug', help='Show error traceback', default=False, action='store_true')
args = parser.parse_args()

URL              = args.url
IMAGE_UPSCALING  = args.sharpen
OPEN_FILE        = args.open
USE_OCR          = not args.no_ocr
PDF_FILE_NAME    = args.output
DEBUG            = args.debug
PAGE_RANGE       = args.pages


print(f"{Back.RED}CourseHeroUnblur CLI - by daijro{Back.RESET}")

log_indic = f"{Fore.RESET}[{Fore.CYAN}>{Fore.RESET}]"
err_indic = f"{Fore.RESET}[{Fore.RED}>{Fore.RESET}]"

log_info = lambda msg: print(f"\n{log_indic} {Fore.CYAN}{msg}{Fore.RESET}")

log_info("Importing libraries...")

import scraper
import webbrowser


try:
    # phase 1: gather website details
    log_info("Getting initial page details...")
    phase1 = scraper.PHASE1(URL, PDF_FILE_NAME)
    phase1.run()

    print(f'''{log_indic} File\'s Info:
    Has data RSID: {phase1.numberDataRsid}
        Data RSID: {phase1.dataRSID}
        Page Amount: {phase1.pageAmount}
        Link Path: {phase1.linkPath}
        File: {phase1.pdf_file_name}''')
    
    # phase 2: scrape and process splits
    log_info("Building pages...")
    phase2 = scraper.PHASE2(
        numberDataRsid  = phase1.numberDataRsid,
        dataRSID        = phase1.dataRSID,
        linkPath        = phase1.linkPath,
        pageAmount      = phase1.pageAmount,
        IMAGE_UPSCALING = IMAGE_UPSCALING,
        PAGE_RANGE      = PAGE_RANGE,
        DEBUG           = DEBUG
    )
    print(Fore.LIGHTBLACK_EX, end="")
    phase2.run(print_eta=True)
    
    # phase 3: write to pdf & ocr
    log_info(f"Writing {'& scanning' if USE_OCR else ''} PDF...")
    phase3 = scraper.PHASE3(
        pdf_file_name = phase1.pdf_file_name,
        USE_OCR       = USE_OCR,
        imgs          = phase2.imgs,
    )
    print(Fore.LIGHTBLACK_EX, end="")
    phase3.run(debug=DEBUG)
    
    # open pdf
    if OPEN_FILE:
        log_info("Opening PDF...")
        webbrowser.open(f"file:///{phase1.pdf_file_name}")
        
    print(f"\n{Fore.RESET}[{Fore.GREEN}>{Fore.RESET}]" + f" {Fore.GREEN}Complete!{Fore.RESET}")

except Exception as e:
    if DEBUG:
        raise e
    print(f"{err_indic} {Fore.RED}{str(e).strip()}\n"
        f"    For more details, run with --debug{Fore.RESET}")
    if e == scraper.PHASE3.exceptions.OCRFailed:
        print(f"{err_indic} {Fore.RED}Skipping OCR...{Fore.RESET}")
    else:
        exit()