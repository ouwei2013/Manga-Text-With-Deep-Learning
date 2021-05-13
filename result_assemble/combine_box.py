"""
Module: combine_box
Desc: combine the resuts of deep learning and the edge-based detection method
Author: Wei Ou
DATE: April 7th 2020
"""

from edge_detector.component_filter import overlap, find_edges, component_filter_by_color
import numpy as np
from edge_detector import clean_page as clean, connected_components as cc


def merge_components(components):
    """Merge a list of overlapping boxes into a single box

    """
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for tmp in components:
        x_mins.append(tmp[1].start)
        x_maxs.append(tmp[1].stop)
        y_mins.append(tmp[0].start)
        y_maxs.append(tmp[0].stop)
    return (slice(min(y_mins), max(y_maxs)), slice(min(x_mins), max(x_maxs)))


def component_shrink(component, shrink_scale=0.15):
    """Function to shrink a box

    """
    w = component[1].stop - component[1].start
    h = component[0].stop - component[0].start
    w_reduced = int(shrink_scale * w / 2)
    h_reduced = int(shrink_scale * h / 2)
    return (slice(component[0].start + h_reduced, component[0].stop - h_reduced),
            slice(component[1].start + w_reduced, component[1].stop - w_reduced))


def edge_cluster_component(edges):
    """Function to compute the box of an edge cluster

    """
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for edge in edges:
        x, y, w, h = edge
        x_mins.append(x)
        x_maxs.append(x + w)
        y_mins.append(y)
        y_maxs.append(y + h)
    return (slice(min(y_mins), max(y_maxs)), slice(min(x_mins), max(x_maxs)))


def smaller_contained_by_larger(component1, component2):
    """Function to check a smaller box is located in a larger box
    """
    x_start = component1[1].start
    x_stop = component1[1].stop
    y_start = component1[0].start
    y_stop = component1[0].stop
    if x_start >= component2[1].start and x_stop <= component2[1].stop and y_start >= component2[0].start and y_stop <= \
            component2[0].stop:
        return True
    else:
        return False


def assemble(img, dl_box, dl_class, class_scores, non_dl_boxes, edges, max_w, max_h, white_background):
    """Function to combine the results of deep learning and the edge-based method

    Args:
        img: the gray-scale image being processed
        dl_box: the detected boxes by the deep learning method
        dl_class: the class label of the detected boxes (bubble or text )
        class_scores: the confidence score of the class label
        non_dl_boxes: the detected boxes by the edge-based method
        edges: the detected edge areas
        white_background: whether the bubble has white background


    Returns:
        Text-like boxes

    """

    ###### keep only the deep learning boxes whose confidence scores are >0.5 ######
    tmp_dl_box = []
    tmp_dl_clas = []
    tmp_dl_scores = []
    for (bx, cls, score) in zip(dl_box, dl_class, class_scores):
        if bx[1] - bx[0] > 0 and bx[3] - bx[2] > 0 and score >= 0.5:
            tmp_dl_box.append(bx)
            tmp_dl_clas.append(cls)
            tmp_dl_scores.append(score)
    dl_box = tmp_dl_box
    dl_class = tmp_dl_clas
    class_scores = tmp_dl_scores
    # print(tmp_dl_scores)
    dl_box = [(slice(box[0], box[1]), slice(box[2], box[3])) for box in dl_box]
    dl_bubbles = [dl_box[idx] for (idx, cls) in enumerate(dl_class) if cls == 1]
    bubble_scores = [class_scores[idx] for (idx, cls) in enumerate(dl_class) if cls == 1]
    dl_texts = [dl_box[idx] for (idx, cls) in enumerate(dl_class) if cls == 2]
    ###### if the text should have white background, then filter out some boxes by colors ########
    if white_background:
        dl_texts = component_filter_by_color(dl_texts, img)

    ##### if a deep learning text box located in a bubble box ,then put them in bubble_text_pair####
    bubble_text_pair = []
    processed_text = []
    processed_bubble = []
    for (i, bubble) in enumerate(dl_bubbles):
        contained_txt = []
        overlapped = False
        for (j, text) in enumerate(dl_texts):
            if smaller_contained_by_larger(text, bubble):
                contained_txt.append(text)
                processed_text.append(j)
                processed_bubble.append(i)
                overlapped = True
        if overlapped:
            bubble_text_pair.append([bubble, contained_txt])

    ########## list to keep text boxes that are located inside no bubble box ###########
    isolated_text = [dl_txt for (idx, dl_txt) in enumerate(dl_texts) if idx not in processed_text]
    #########  list to keep bubble boxes that have no text boxes located inside ######
    isolated_bubble = [bubble for (idx, bubble) in enumerate(dl_bubbles) if idx not in processed_bubble]
    isolated_bubble_scores = [score for (idx, score) in enumerate(bubble_scores) if idx not in processed_bubble]

    txt_components = []
    for (i, (bubble, txts)) in enumerate(bubble_text_pair):
        txt_components.append(merge_components(txts))

    txt_components = txt_components + isolated_text

    ########## we first process the isolated bubble boxes #########
    bubble_components = []
    for (i, bubble) in enumerate(isolated_bubble):
        bubble = component_shrink(bubble)
        ######## check whether there exist any edge boxes inside the bubble box ########
        edges = find_edges(bubble, edges)
        ######### if no edge exists, keep it only when its confidence score >0.6 ########
        if len(edges) == 0:
            if isolated_bubble_scores[i] > 0.6:
                bubble_components.append(bubble)
        ######## if there exist some edges, compute a tighter box #############
        else:
            bubble_components.append(edge_cluster_component(edges))

    ########## keep only the bubble boxes that do not overlap with any boxes by the edge-based method######
    good_bubble_components = []
    for bubble_bx in bubble_components:
        overlapped = False
        for (idx, nd_bx) in enumerate(non_dl_boxes):
            if overlap(nd_bx, bubble_bx):
                # if cc.area_bb(nd_bx)>cc.area_bb(bubble_bx):
                #     bad_nd_component_ids.append(idx)
                #     good_bubble_components.append(bubble_bx)
                overlapped = True
                break
        if not overlapped:
            good_bubble_components.append(bubble_bx)
    ############### merge overlapping boxes #########################
    mask = np.zeros(img.shape)
    for component in txt_components + non_dl_boxes:
        mask[component[0].start:component[0].stop, component[1].start:component[1].stop] = 1
    txt_components = cc.get_connected_components(mask)

    # // plt.imshow(mask)
    # plt.show()
    # candidate_components = cc.get_connected_components(mask)

    return txt_components, good_bubble_components
