ARG tensorflow_tag

FROM tensorflow/tensorflow:${tensorflow_tag}

RUN pip3 install keras sklearn