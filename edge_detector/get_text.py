"""
Module: get_text
Desc: detect text box by detecting edges
Author: Wei Ou
DATE: April 7th 2020
"""

import os
import cv2
from edge_detector import clean_page as clean, connected_components as cc, segmentation as seg
from edge_detector.component_filter import component_filter
from matplotlib import pyplot as plt


def locate_text(img):
    gray = clean.grayscale(img)
    # binary_threshold=arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
    # inv_binary = cv2.bitwise_not(clean.binarize(gray, threshold=binary_threshold))
    # binary = clean.binarize(gray, threshold=binary_threshold)
    segmented_image, bounding_boxes = seg.segment_image(img.copy(), gray)
    segmented_image = segmented_image[:, :, 2]
    # segmented_image = cv2.bitwise_or(segmented_image,mser_mask)
    components = cc.get_connected_components(segmented_image)
    components, white_txt_background = component_filter(components, gray, bounding_boxes)
    # components=component_filter(components,img,bounding_boxes)
    return components, bounding_boxes, white_txt_background

    # components = component_filter_size(components,gray,ref_hist)

    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    file_path = '../ocr_data/famtaimg0006.jpg'
    img = cv2.imread(file_path)
    components, _, _ = locate_text(img)
    cc.draw_bounding_boxes(img, components, color=(255, 0, 0), line_size=2)
    plt.imshow(img)
    plt.show()
