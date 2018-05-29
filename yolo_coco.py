import argparse
from collections import namedtuple
import json
import os
from PIL import Image
import numpy as np

from yolo import YOLO
from pycocotools.coco import COCO

COCOResultItem = namedtuple('COCOResultItem', ['image_id', 'category_id', 'bbox', 'score'])

img_dirs = {
    'test-dev2017': 'test2017',
    'val2017': 'val2017',
}

anno_names = {
    'test-dev2017': 'image_info_test-dev2017.json',
    'val2017': 'instances_val2017.json',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dir', type=str, default='/home/cwq/data/COCO/2017')
    parser.add_argument('--dataset', type=str, default='test-dev2017', choices=['test-dev2017', 'val2017'])

    flags, _ = parser.parse_known_args()
    flags.img_dir = os.path.join(flags.coco_dir, img_dirs[flags.dataset])
    flags.anno_file = os.path.join(flags.coco_dir, 'annotations', anno_names[flags.dataset])
    return flags


if __name__ == '__main__':
    """
    Run yolo on coco test-dev, and save result file.
    """
    flags = parse_args()
    coco = COCO(flags.anno_file)

    img_paths = []
    img_ids = coco.getImgIds()
    for i in img_ids:
        img_paths.append(os.path.join(flags.img_dir, '%.12d.jpg' % i))

    category_ids = []
    for item in coco.dataset['categories']:
        category_ids.append(item['id'])

    results = []
    yolo = YOLO()
    for j, img_path in enumerate(img_paths):
        img_id = img_ids[j]
        image = Image.open(img_path)
        _, boxes, scores, classes = yolo.detect_image(image, draw=False)

        for i in range(len(boxes)):
            top, left, bottom, right = boxes[i]
            box = [float(left), float(top), float(right - left), float(bottom - top)]

            category_id = category_ids[classes[i]]
            r = COCOResultItem(img_id, category_id, box, float(scores[i]))
            results.append(r._asdict())

        print("%d/%d" % (j, len(img_paths)), end='\r')

    yolo.close_session()

    file_path = 'coco/result/detections_{}_kyolov3_results.json'.format(flags.dateset)
    with open(file_path, 'w') as f:
        json.dump(results, f)
