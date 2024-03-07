# https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow

# We expect src/data to be
VOLUME /src
VOLUME /data

CMD python ai/hello.py

