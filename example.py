import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from create_bounding_box import load_image_into_numpy_array, run_inference_for_single_image
from edge_detector import connected_components as cc
from edge_detector.get_text import locate_text
# from edge_detector.mser import overlap, find_edges, expand_component_1
from object_detection.utils import label_map_util
from text_detector import initializer, detect_chars_rect

graph = initializer()
data_path = 'ocr_data/'
save_path = 'test_result/'
files = os.listdir(data_path)
for file in files:
    if file.endswith('jpg') or file.endswith('png') or file.endswith('PNG'):
        file_path = os.path.join(data_path, file)
        img, components = detect_chars_rect(graph, file_path)
        cc.draw_bounding_boxes(img, components, color=(255, 0, 0), line_size=2)
        result_save_path = os.path.join(save_path, file)
        cv2.imwrite(result_save_path, img)
        print('%s has been processed ' % file)
