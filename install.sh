#!/usr/bin/env bash

command -v python3 >/dev/null 2>&1 || {
    echo >&2 "Please install python3 to use this script."
    exit 1
}

if [ ! -d data ]; then
    mkdir data
fi
cd data
if [ ! -d FDDB-folds ]; then
    echo 'Downloading the FDDB face annotations...'
    curl -O 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
    tar -xvzf FDDB-folds.tgz
    rm FDDB-folds.tgz
fi
if [ ! -d FDDB-images ]; then
    echo 'Downloading the FDDB images...'
    curl -O 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
    mkdir FDDB-images
    tar -xvzf originalPics.tar.gz -C FDDB-images
    rm originalPics.tar.gz
fi
if [ ! -f opencv_face_detector_uint8.pb ]; then
    echo 'Downloading TensorFlow pretrained model weights from OpenCV...'
    curl -O 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb'
fi
if [ ! -f opencv_face_detector.pbtxt ]; then
    echo 'Downloading TensorFlow model file from OpenCV...'
    curl -O 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt'
fi
if [ ! -f haarcascade_frontalface_default.xml ]; then
    echo 'Downloading Haar cascade pretrained model weights from OpenCV...'
    curl -O 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
fi
cd ..

if [ ! -d env ]; then
    python3 -m venv env
fi
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
