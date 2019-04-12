A script to compare OpenCV's face detection methods.

## Background

OpenCV has been around for a long time. The original EZ face detection was the classic Haar cascade classifier (also known as the Viola-Jones detector), and for many years OpenCV has shipped with pretrained model parameters to plug and play. But since then, deep neural networks (DNN) have made advances in the state of the art, and now OpenCV also has a pretrained model for a DNN face detector.

This script shows how these detectors perform on the FDDB dataset. It displays the average precision for different scenarios as well as the precision-recall curves for each detector, as provided by the COCO protocol. It also shows the average detection time per image on your machine.

## Results

The DNN is more accurate, and just a little slower. Run it yourself and see!

## Usage

```
# Download the OpenCV models and FDDB dataset, set up a python virtualenv, and install python requirements.
./install.sh
# Activate the virtualenv and run compare_face_detection.py.
./run.sh
```
