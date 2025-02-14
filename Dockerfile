FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

RUN pip install tensorflow
RUN pip install git+https://github.com/huggingface/transformers accelerate
RUN pip install runpod
RUN pip install qwen-vl-utils

ADD src .

CMD python3.11 -u /handler.py