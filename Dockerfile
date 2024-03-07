# https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow:2.15.0

RUN apt-get update
RUN apt-get install -y libasound-dev portaudio19-dev libgl1-mesa-glx tesseract-ocr ffmpeg

COPY docker-requirements.txt docker-requirements.txt
RUN pip install -r docker-requirements.txt

## Copy expected source code + data into the container
#COPY src /src
#COPY data /data
COPY data/tesseract /data/tesseract

# We expect src/data to be
VOLUME /src
VOLUME /data
VOLUME /data-series
VOLUME /input-data-folder

CMD python /src/extractor.py

