import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from create_bounding_box import load_image_into_numpy_array, run_inference_for_single_image
from edge_detector import connected_components as cc
from edge_detector.get_text import locate_text
from object_detection.utils import label_map_util
from edge_detector.component_filter import expand_component
from result_assemble.combine_box import assemble, merge_components
from result_assemble.clean_components import merge_overlapped
from edge_detector.mser import mser
import edge_detector.clean_page as clean


def initializer():
    """Function to initialize  tensorflow graph for the deep learning technique

    Returns:
        Neural network computation graphs

    """
    PATH_TO_FROZEN_GRAPH = 'detector/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('detector', 'balao.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def detect_chars_rect(detection_graph, file_path):
    """Function to detect text in a manga image

    Args:
        detection_graph: tensorflow graph
        file_path: the path of the image file of interest

    Returns:
        all_components: all text boxes in format (slice (y_start,y_stop),slice (x_start,x_stop))

    """
    # img = cv2.imread(file_path)
    image = Image.open(file_path)
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    image_np = load_image_into_numpy_array(image)
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    decimal_boxes = output_dict['detection_boxes']
    box_classes = output_dict['detection_classes']
    class_scores = output_dict['detection_scores']
    dl_boxes_large = [[int(box[0] * im_height), int(box[2] * im_height),
                       int(box[1] * im_width), int(box[3] * im_width)] for box in decimal_boxes]
    # dl_boxes_txt = [dl_boxes_large[idx] for idx, cls in enumerate(box_classes) if cls == 2]
    # dl_boxes_txt = [(slice(box[0], box[1]), slice(box[2], box[3])) for box in dl_boxes_txt]
    # dl_boxes_bubble = [dl_boxes_large[idx] for idx, cls in enumerate(box_classes) if cls == 1]
    # dl_boxes_bubble = [(slice(box[0], box[1]), slice(box[2], box[3])) for box in dl_boxes_bubble]
    img = cv2.imread(file_path)
    non_dl_boxes, edges, white_background = locate_text(img)
    # print('non_dl_box len %d'%len(non_dl_boxes))
    txt_boxes, bubble_boxes = assemble(img, dl_boxes_large, box_classes, class_scores, non_dl_boxes, edges,
                                       img.shape[1] * 0.4, img.shape[0] * 0.4, white_background)
    # print('combined box len %d' %len(components))
    txt_boxes = expand_component(img[:, :, 1], txt_boxes, edges, rlso_h=20, rlso_w=20, expand_h=10, expand_w=10)
    all_componets = merge_overlapped(txt_boxes + bubble_boxes)

    ######## if the edge-based technique and deep learning technique give no result, then bring in the mser technique ##
    if len(all_componets) < 2:
        component_width = all_componets[0][0].stop - all_componets[0][0].start
        component_height = all_componets[0][1].stop - all_componets[0][1].start
        min_side = min(component_height, component_width)
        if min_side < 50:
            all_componets = mser(clean.grayscale(img))

    # non_dl_boxes=expand_component(img[:,:,1],non_dl_boxes,edges,rlso_w=20,expand_h=20,expand_w=20)
    return img, all_componets  # +non_dl_boxes
    # img_tmp=img.copy()
    # img_tmp1=img.copy()
    # cc.draw_bounding_boxes(img,dl_boxes+non_dl_boxes,color=(255,0,0),line_size=2)
    # cc.draw_bounding_boxes(img_tmp,dl_boxes_txt,color=(255,0,0),line_size=2)
    # cc.draw_bounding_boxes(img_tmp1,dl_boxes_bubble,color=(255,0,0),line_size=2)
    # cc.draw_bounding_boxes(img,dl_boxes_tight,color=(0,255,0),line_size=5)
    # ax1=plt.subplot(1,3,1)
    # ax1.imshow(img)
    # ax2=plt.subplot(1,3,2)
    # ax2.imshow(img_tmp)
    # ax3=plt.subplot(1,3,3)
    # ax3.imshow(img_tmp1)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Please enter an image path')
    args = parser.parse_args()
    file_path = args.infile
    graph = initializer()
    img, components = detect_chars_rect(graph, file_path)
    cc.draw_bounding_boxes(img, components, color=(255, 0, 0), line_size=2)
    plt.imshow(img)
    plt.show()



# image = cv2.imread('../manga_text_detector/ocr_data/img0080.jpg')
# file_path= '../manga_text_detector/ocr_data/hinotori_1.png'
# image = Image.open(file_path )
# image = image.convert('RGB')
# (im_width, im_height) = image.size
# image_np = load_image_into_numpy_array(image)
# image_np_expanded = np.expand_dims(image_np, axis=0)
# output_dict = run_inference_for_single_image(image_np, detection_graph)
# decimal_boxes = output_dict['detection_boxes']
# dl_boxes_large = [[int(box[0]*im_height),int(box[2]*im_height),
#              int(box[1]*im_width),int(box[3]*im_width)] for box in  decimal_boxes]

# img = cv2.imread(file_path)
# non_dl_boxes,edges = locate_text(img)
# dl_boxes_tight =[]
# for box in dl_boxes_large:
#     box_slice=(slice(box[0],box[1]),slice(box[2],box[3]))
#     box_edges=find_edges(box_slice,edges)
#     if len(box_edges)>0:
#         edge_x_min= min([be[0] for be in box_edges])
#         edge_y_min = min([be[1] for be in box_edges])
#         edge_x_max = max([be[0]+be[2] for be in box_edges])
#         edge_y_max = max([be[1]+be[3]for be in box_edges])
#         dl_boxes_tight.append((slice(edge_y_min,edge_y_max),slice(edge_x_min,edge_x_max)))
#     else:
#         dl_boxes_tight.append(box_slice)

# cleaned_non_dl_boxes=[]
# for non_dl_bx in non_dl_boxes:
#     keep_flag=True
#     for dl_bx in dl_boxes_tight:
#         if overlap(dl_bx,non_dl_bx):
#             keep_flag=False
#             break
#     if keep_flag:cleaned_non_dl_boxes.append(non_dl_bx)
#
# components = cleaned_non_dl_boxes+dl_boxes_tight
# print(components)
# cc.draw_bounding_boxes(img,components,color=(255,0,0),line_size=2)
# plt.imshow(img)
# plt.show()











# print(output_dict['detection_boxes'])

# visualization_utils.visualize_boxes_and_labels_on_image_array(
#           image_np,
#           output_dict['detection_boxes'],
#           output_dict['detection_classes'],
#           output_dict['detection_scores'],
#           category_index,
#           instance_masks=output_dict.get('detection_masks'),
#           use_normalized_coordinates=True,
#           line_thickness=8)




# plt.figure(figsize=(19, 16))
# plt.imshow(image_np)
# plt.show()
