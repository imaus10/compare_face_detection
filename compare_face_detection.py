import cv2
from ellipse2rect import ellipse2rect
from fnmatch import fnmatch
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from time import time
from tqdm import tqdm

def get_detectors():
    '''
    Contructs the DNN and Haar cascade detectors.
    Add another detection method here and it will also be evaluated.
    '''
    # DNN
    net = cv2.dnn.readNet('data/opencv_face_detector_uint8.pb', 'data/opencv_face_detector.pbtxt')
    def dnn_detect(img, img_id):
        net.setInput(cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104., 177., 123.), False, False))
        out = net.forward()
        detections = []
        for i in range(out.shape[2]):
            confidence = float(out[0, 0, i, 2])
            left = int(out[0, 0, i, 3] * img.shape[1])
            top = int(out[0, 0, i, 4] * img.shape[0])
            right = int(out[0, 0, i, 5] * img.shape[1])
            bottom = int(out[0, 0, i, 6] * img.shape[0])
            width = right - left
            height = bottom - top
            detections.append({
                'image_id'    : img_id,
                'category_id' : 0,
                'bbox'        : [left, top, width, height],
                'score'       : confidence
            })
        return detections

    # Haar cascade
    cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    def haar_cascade_detect(img, img_id):
        detections = []
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(img_gray)
        for rect in rects:
            left, top, width, height = list(map(int, rect))
            detections.append({
                'image_id'    : img_id,
                'category_id' : 0,
                'bbox'        : [left, top, width, height],
                'score'       : 1
            })
        return detections

    return {
        'DNN'          : dnn_detect,
        'haar_cascade' : haar_cascade_detect
    }

def get_fddb_dataset():
    '''Read the FDDB annotations in COCO format'''
    print('Loading FDDB dataset...')
    dataset = {
        # COCO is for general object detection, but we're only interested in faces
        'categories'  : [{ 'id': 0, 'name': 'face' }],
        'images'      : [],
        'annotations' : []
    }
    annotation_dir = 'data/FDDB-folds'
    img_id = 0
    for annotation_filename in os.listdir(annotation_dir):
        if fnmatch(annotation_filename, 'FDDB-fold-*-ellipseList.txt'):
            with open(os.path.join(annotation_dir, annotation_filename)) as f:
                line = f.readline().strip()
                while line:
                    img_filename = '{}.jpg'.format(os.path.join('data/FDDB-images', line))
                    img_id = len(dataset['images'])
                    dataset['images'].append({
                        'id'        : img_id,
                        'file_name' : img_filename
                    })
                    num_faces = int(f.readline().strip())
                    for i in range(num_faces):
                        ellipse_params = list(map(float, f.readline().strip().split(' ')[:5]))
                        left, top, width, height = ellipse2rect(*ellipse_params)
                        dataset['annotations'].append({
                            'id'          : len(dataset['annotations']),
                            'image_id'    : img_id,
                            'category_id' : 0,
                            'bbox'        : [left, top, width, height],
                            'iscrowd'     : 0,
                            'area'        : width * height
                        })
                    line = f.readline().strip()
                    img_id += 1
    return dataset

def get_detections(imgs, detectors):
    '''Run the detectors on the given images'''
    print('Gathering detections...')
    detections = { detector_name: [] for detector_name in detectors.keys() }
    meta_info = {
        'img_size'     : [],
        'detect_times' : { detector_name: [] for detector_name in detectors.keys() },
    }
    for img_info in tqdm(imgs):
        img_id = img_info['id']
        filename = img_info['file_name']
        img = cv2.imread(filename)
        meta_info['img_size'].append(list(map(int, img.shape[:2])))
        for detector_name, detect in detectors.items():
            start = time()
            detections[detector_name].extend(detect(img, img_id))
            detect_time = time() - start
            meta_info['detect_times'][detector_name].append(detect_time)
    with open('data/meta.json', 'w') as f:
        json.dump(meta_info, f)
    for detector_name, detector_detections in detections.items():
        with open('data/detections_{}.json'.format(detector_name), 'w') as f:
            json.dump(detector_detections, f)

def plot_meta_info():
    '''Plots on the dataset images and detection time efficiency'''
    with open('data/meta.json') as f:
        meta_info = json.load(f)
    fig = plt.figure()
    plt.title('Histogram of image size in FDDB')
    pixel_counts = np.prod(meta_info['img_size'], axis=1)
    _, bins, _ = plt.hist(pixel_counts, edgecolor='black')
    plt.xlabel('Pixel count')
    x_tick_labels = [ '{0}x{0}'.format(int(math.sqrt(bin))) for bin in bins ]
    plt.xticks(bins, x_tick_labels)
    fig.autofmt_xdate()
    plt.ylabel('Number of images')
    plt.show()

    fig = plt.figure()
    plt.title('Mean detect time per image in FDDB (s)')
    x_tick_labels = []
    mean_detect_times = []
    for detector_name, detect_times in meta_info['detect_times'].items():
        x_tick_labels.append(detector_name)
        mean_detect_times.append(np.mean(detect_times))
    x = np.arange(len(mean_detect_times))
    plt.bar(x, mean_detect_times, tick_label=x_tick_labels, edgecolor='black')
    plt.show()

def evaluate(dataset, detector_names):
    '''Evaluates the performance of each method using the COCO protocol'''
    plot_meta_info()

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = dataset
    coco_ground_truth.createIndex()
    agg_stats = []
    precision = []
    recall = []
    for index, detector_name in enumerate(detector_names):
        detections_filename = 'data/detections_{}.json'.format(detector_name)
        coco_detections = coco_ground_truth.loadRes(detections_filename)
        coco_eval = COCOeval(coco_ground_truth, coco_detections, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        # Get the precision values for IoU@0.5, at all scales, with 100 max detections per image
        precision.append(coco_eval.eval['precision'][0,:,0,0,2])
        recall.append(coco_eval.params.recThrs)
        coco_eval.summarize()
        agg_stats.append(coco_eval.stats[:6])

    fig, ax = plt.subplots()
    ax.set_title('Precision scores on FDDB')
    # Show just the first 6 metrics
    x = np.arange(6)
    ax.set_xticklabels((
        'AP @ IoU 0.5-0.95',
        'AP @ 0.5 IoU',
        'AP @ 0.75 IoU (strict)',
        'AP small',
        'AP medium',
        'AP large'
    ))
    # This width might not work for more than two detectors?
    bar_width = 0.35
    ax.set_xticks(x + bar_width / 2)
    plot_groups = [
        ax.bar(x + (index * bar_width), agg_stats[index], bar_width)
        for index in range(len(detector_names))
    ]
    ax.legend(plot_groups, detector_names)
    fig.autofmt_xdate()
    plt.show()

    fig = plt.figure()
    plt.xlabel('Recall')
    plt.xlim((0, 1))
    plt.ylabel('Precision')
    plt.ylim((0, 1))
    plot_groups = [ plt.plot(r, p) for r, p in zip(recall, precision) ]
    plt.legend(detector_names)
    plt.show()

if __name__ == '__main__':
    dataset = get_fddb_dataset()
    detectors = get_detectors()
    # get_detections(dataset['images'], detectors)
    evaluate(dataset, detectors.keys())
