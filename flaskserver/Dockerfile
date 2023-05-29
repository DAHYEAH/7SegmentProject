FROM python:3.8

RUN pip3 install flask
RUN pip3 install flask_cors
RUN pip3 install lmdb
RUN pip3 install pillow
RUN pip3 install torchvision
RUN pip3 install nltk
RUN pip3 install natsort
RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python
RUN pip3 install opencv-python-headless

COPY . /workspace

WORKDIR /workspace

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
















