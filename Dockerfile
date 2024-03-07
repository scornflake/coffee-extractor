# https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow:2.15.0

RUN apt-get update
RUN apt-get install -y libasound-dev portaudio19-dev libgl1-mesa-glx tesseract-ocr ffmpeg

COPY docker-requirements.txt docker-requirements.txt
RUN pip install -r docker-requirements.txt

## Copy expected source code + data into the container
#COPY src /src
#COPY data /data
COPY data/tesseract /data/teseract

# We expect src/data to be
VOLUME /src
VOLUME /data
VOLUME /data-series
VOLUME /input-data-folder

# Args: --audio --video input.spec.json
# Working: /Users/neil/Documents/Coffee/2

#CMD python /src/ai/hello.py

CMD python /src/extractor.py

