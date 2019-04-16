A script to compare OpenCV's face detection methods.

## Background

OpenCV has been around for a long time. The original EZ face detection was the classic Haar cascade classifier (also known as the Viola-Jones detector), and for many years OpenCV has shipped with pretrained model parameters to plug and play. But since then, deep neural networks (DNN) have made advances in the state of the art, and now OpenCV also has a pretrained model for a DNN face detector.

This script shows how these detectors perform on the FDDB dataset. It displays the average precision for different scenarios as well as the precision-recall curves for each detector, as provided by the COCO protocol. It also shows the average detection time per image on your machine.

## Results

The DNN is more accurate, and just a little slower. Run it yourself and see! Here are the results on my machine:

![FDDB image size histogram](/results/01_FDDB_image_size_histogram.png?raw=true)
![Mean detect time](/results/02_mean_detect_time.png?raw=true)
![Average precision scores](/results/03_average_precision_scores.png?raw=true)
![PR curves](/results/04_PR_curves.png?raw=true)

## Usage

```
# Download the OpenCV models and FDDB dataset, set up a python virtualenv, and install python requirements.
./install.sh
# Activate the virtualenv and run compare_face_detection.py.
./run.sh
```

## References

- [Viola-Jones cascade classifier](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- [SSD object detection (DNN)](https://arxiv.org/abs/1512.02325)
