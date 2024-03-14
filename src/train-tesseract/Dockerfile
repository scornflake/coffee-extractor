FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y tesseract-ocr

VOLUME /tessdata
VOLUME /fonts
VOLUME /output

CMD text2image --fonts_dir /fonts --list_available_fonts
