ARG tensorflow_tag

FROM tensorflow/tensorflow:${tensorflow_tag}

RUN apt-get update && \
  apt-get install -y python3-tk nano && \
  pip3 install keras keras-tqdm matplotlib sklearn