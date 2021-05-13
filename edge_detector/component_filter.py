"""
Module: component_filter
Desc: filter out non-text boxes
Author: Wei Ou
DATE: April 7th 2020
"""

import cv2
import numpy as np
# import scipy.ndimage
# import skimage.morphology
from skimage.feature import greycomatrix, greycoprops
import edge_detector.defaults as  defaults
from edge_detector import clean_page as clean, connected_components as cc
from edge_detector.run_length_smoothing import RLSO
from matplotlib import pyplot as plt


def compute_contrast(aoi):
    """Function to compute the contrast of a given area

    Args:
        aoi: a gray-scale aoi


    Returns:
        The contrast of the given aoi

    """
    glcm = greycomatrix(aoi, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return (greycoprops(glcm, 'dissimilarity')[0, 0])


def find_edges(text_box, edge_boxes):
    """Finding edge-enclosed areas in a given box.
       purpose: edge-enclosed areas are very likely to be text, therefore
       if text_box contains edge areas,it is very likely to be a good text box candidate
    Args:
        text_box: The given text box
        edge_boxes: All detected edge boxes in the image

    Returns:
        The (x,y,w,h) of all edge-enclosed areas located in text_box

    """
    selected_edge_boxes = []
    for box in edge_boxes:
        x, y, w, h = box
        if text_box[0].start <= y and text_box[0].stop >= y + h and text_box[1].start <= x and text_box[
            1].stop >= x + w:
            selected_edge_boxes.append(box)
    return selected_edge_boxes


def color_hist_single(aoi):
    """ Compute the gray-scle histogram of a given aoi

    Args:
        aoi: gray-scale aoi

    Returns:
        30-dimensional histogram vector

    """
    hist = cv2.calcHist([aoi], [0], None, [30], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def color_hist_block(aois):
    """ Compute the gray-scle histogram of a list of aois

    """
    hist = cv2.calcHist([aois[0]], [0], None, [30], [0, 256])
    contrasts = []
    for aoi in aois[1:]:
        hist = hist + cv2.calcHist([aoi], [0], None, [30], [0, 256])
        contrasts.append(compute_contrast(aoi))
    hist = cv2.normalize(hist, hist).flatten()
    # print hist,contrasts
    return hist


def hist_sim(hist1, hist2):
    """ Compute the similarity between two hist vectors

    """
    # print(np.dot(hist1,hist2.T)/(np.linalg.norm(hist1)*np.linalg.norm(hist2)))
    sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # print('sim')
    return sim


def edge_cluster_rectangle(cluster):
    """ Compute the box coordinate for an edge cluster

    Args:
        cluster: a list of edge boxes

    Returns:
        Box in (x_min,y_min,x_max,y_max) format

    """
    x_starts = []
    x_stops = []
    y_starts = []
    y_stops = []
    for area in cluster:
        x, y, w, h = area
        x_starts.append(x)
        x_stops.append(x + w)
        y_starts.append(y)
        y_stops.append(y + h)
    return min(x_starts), min(y_starts), max(x_stops) - min(x_starts), max(y_stops) - min(y_starts)


def border_analyze_vertical(mask, borders, min_segment_width=5, max_border_width=0.4):
    """ A candidate text box can be segmented (in vertical direction) by thin lines into small blocks. The resulting
    blocks should have similar widths or heights.

    Args:
        mask: mask resulting from the edge-based segmentation
        borders: the coordinates of borders between two neighboring blocks
        min_segment_width: the minimum width of a resulting block
        max_border_width: the maximum width scale of a resulting block

    Returns:
        True (if the resulting blocks have similar widths or heights), False otherwise

    """
    max_border_width = mask.shape[1] * max_border_width
    if len(borders) == 0: return False
    left_most = 0
    right_most = mask.shape[1] - 1
    # middle_borders = [border for border in borders if (left_most in border or right_most in border) and abs(border[0]-border[1])>3  ]
    # if len(middle_borders)==0:return False
    slice_start = 0
    # segs_by_middle_borders=[]
    block_widths = []
    for v_border in borders[::-1]:
        border_start = v_border[1]
        border_stop = v_border[0]
        if border_start == left_most:
            slice_start = border_stop
        elif border_stop == right_most:
            # mask_slice=mask[slice_start:border_start]
            block_widths.append(border_start - slice_start)
            # segs_by_middle_borders.appen(mask_slice)
            slice_start = border_stop
        elif border_stop - border_start >= 1:
            # mask_slice = mask[slice_start,border_start]
            if border_stop - border_start < max_border_width:
                block_widths.append(border_start - slice_start)
            # segs_by_middle_borders.appen(mask_slice)
            slice_start = border_stop
    if slice_start < right_most:
        block_widths.append(mask.shape[1] - slice_start)
        # segs_by_middle_borders.append(mask[slice_start:])

    block_widths = [w for w in block_widths if w > min_segment_width]
    # print('block widths')
    # print(block_widths)
    # print('block widths deviation' )
    # print(np.std(block_widths))
    # print('block heights dev/mean' )
    # print(np.std(block_widths)/np.mean(block_widths))

    if len(block_widths) < 2: return False
    return segment_analyzer(block_widths)


def border_analyze_horizon(mask, borders, min_segment_height=5, max_border_height=0.4):
    """ A candidate text box can be segmented (in horizontal direction) by thin lines into small blocks. The resulting
    blocks should have similar widths or heights.

    Args:
        mask: mask resulting from the edge-based segmentation
        borders: the coordinates of borders between two neighboring blocks
        min_segment_height: the minimum height of a resulting block
        max_border_heightt: the maximum height scale of a resulting block


    Returns:
        True (if the resulting blocks have similar widths or heights), False otherwise

    """
    max_border_height = mask.shape[0] * max_border_height
    if len(borders) == 0: return False
    upper_most = 0
    lower_most = mask.shape[0] - 1
    # middle_borders = [border for border in borders if (left_most in border or right_most in border) and abs(border[0]-border[1])>3  ]
    # if len(middle_borders)==0:return False
    slice_start = 0
    block_heights = []
    block_areas = []
    # segs_by_middle_borders=[]
    for v_border in borders:
        border_start = v_border[0]
        border_stop = v_border[1]
        if border_start == upper_most:
            slice_start = border_stop
        elif border_stop == lower_most:
            # mask_slice=mask[slice_start:border_start]
            block_heights.append(border_start - slice_start)

            block_areas.append(np.sum(mask[slice_start:border_start, :]))
            # segs_by_middle_borders.append(mask_slice)
            slice_start = border_stop
        elif border_stop - border_start >= 1:
            # mask_slice = mask[slice_start,border_start]
            if border_stop - border_start < max_border_height:
                block_heights.append(border_start - slice_start)
                block_areas.append(np.sum(mask[slice_start:border_start, :]))
            # segs_by_middle_borders.append(mask_slice)
            slice_start = border_stop
    if slice_start < lower_most:
        # segs_by_middle_borders.append(mask[slice_start:])
        block_heights.append(mask.shape[0] - slice_start)
        block_areas.append(np.sum(mask[slice_start:mask.shape[0], :]))

    if len(block_areas) < 2:
        return False
    if len(block_areas) == 2:
        max_area = max(block_areas)
        min_area = min(block_areas)
        if max_area / min_area > 3:
            return False
    if len(block_areas) > 2:
        if np.std(block_areas) / np.mean(block_areas) > 2:
            return False

    # print(block_areas)
    # print(mask.shape[0]*mask.shape[1])
    # print(np.std(block_areas)/np.mean(block_areas))
    block_heights = [h for h in block_heights if h > min_segment_height]

    # print('block areas ')
    # print(block_areas)

    # print('block heights deviation' )
    # print(np.std(block_heights))

    # print('block heights dev/mean' )
    # print(np.std(block_heights)/np.mean(block_heights))

    if len(block_heights) < 2: return False
    return segment_analyzer(block_heights)


def density_analysis(mask):
    """ A good text box should have a certain amount of white colors and black colors
    If a box contains only white colors or only black colors, then it is very likely not to be a text box
    This function first divides the mask into smaller blocks, then compute the intensity ratios between white and
    black.

    Args:
        mask: mask resulting from the edge-based segmentation

    Returns:
        The minimum white/black ratio among the small blocks

    """
    if mask.shape[0] < 10 or mask.shape[1] < 10: return 1.0
    width = mask.shape[1]
    # print('width %d'%width)
    sub_widths = np.linspace(0, width, 4)
    zero_windows = []
    for i in range(len(sub_widths) - 1):
        sub_mask = mask[:, int(sub_widths[i]):int(sub_widths[i + 1])]
        windowsize_r = 3
        windowsize_c = 3
        z_window = 0
        total_window = 0
        for r in range(0, sub_mask.shape[0] - windowsize_r, windowsize_r):
            for c in range(0, sub_mask.shape[1] - windowsize_c, windowsize_c):
                window = sub_mask[r:r + windowsize_r, c:c + windowsize_c]
                total_window = total_window + 1
                if np.sum(window) == 0: z_window = z_window + 1
        if total_window > 0:
            zero_windows.append(z_window / total_window)
    return min(zero_windows)


def one_dimension_val_clutering(vals, max_distance=5):
    """ One-dimension clustering on segment widths or heights
    If a list of segments share similar widths or heights,
    these segments will fall in the same clusters

    Args:
        vals: segment widths or heights
        max_distance: the max distance between two neighboring values

    Returns:
        Clusters of segments share similar widths or heights

    """
    vals = sorted(vals)
    clusters = []
    for (idx, i) in enumerate(vals):
        cluster = [j for j in vals if abs(j - i) < max_distance]
        clusters.append(cluster)
    clusters = sorted(clusters, key=len, reverse=True)
    cluster = clusters[0]
    if len(cluster) / len(vals) > 0.6 or len(cluster) >= 3:
        return cluster
    else:
        return []


def segment_analyzer(sizes, cluster_size_std=defaults.MAX_SEGMENT_CLUSTER_SIZE_STD):
    """ Function to analyze the segment widths or heights to decide whether it is a good candidate
    Args:
        sizes: width or height value clusters
        cluster_size_std: the maximum allowed standard deviation of a width or height cluster
    Returns:
        True if the size clusters look good, False otherwise
    """
    if len(sizes) < 2: return False
    size_std = np.std(sizes)
    size_mean = np.mean(sizes)
    if len(sizes) == 2:
        if size_std <= cluster_size_std and size_std / size_mean < 0.2:
            return True
        else:
            return False
    if size_std <= cluster_size_std and size_std / size_mean <= 0.3:
        return True
    cluster = one_dimension_val_clutering(sizes)
    if len(cluster) == 0: return False
    cluster_std = np.std(cluster)
    cluster_mean = np.mean(cluster)
    if len(cluster) == 2:
        if cluster_std / cluster_mean < 0.1:
            return True
        else:
            return False
    if cluster_std / cluster_mean < 0.2 or len(cluster) >= 4:
        return True
    return False


def edge_not_in_component(edge, component):
    """ Function to check whether an edge area is in a large text box
    Args:
        edge: coordinate of an edge box in the (x,y,w,h) format
        component: a larger candidate box
    Returns:
        True or False
    """
    x_start = edge[0]
    x_stop = edge[0] + edge[2]
    y_start = edge[1]
    y_stop = edge[1] + edge[3]
    if x_start >= component[1].start and x_stop <= component[1].stop and y_start >= component[0].start and y_stop <= \
            component[0].stop:
        return False
    else:
        return True


def overlap(component1, component2):
    """ Function to check whether two boxes overlap
    Returns:
        True or False
    """
    if component1[0].start <= component2[0].stop and component2[0].start <= component1[0].stop:
        if component1[1].start <= component2[1].stop and component2[1].start <= component1[1].stop:
            return True
    return False


def expand_component(img, components, edges, rlso_h=10, rlso_w=10, expand_h=20, expand_w=20,
                     max_allowed_long_side=defaults.EXPAND_MAX_LONGER_SIDE,
                     max_allowed_short_side=defaults.EXPAND_MAX_SHORTER_SIDE):
    """ Function to expand a candidate text box in order to include all characters inside
    a conversation bubble into a signle box

    Args:
        img: the gray-scale image of interest
        components: all candidate text boxes
        edges: all detected edge boxes
        rlso_h: run-length-smoothing-operation height
        rlso_w: run-length-smoothing-operation width
        expand_h: the maximum length a component is allowed to expand in the vertical direction

    Returns:
        expanded text boxes
    """

    mask = np.zeros(img.shape)
    for edge in edges:
        if all([edge_not_in_component(edge, tmp) for tmp in components]):
            mask[edge[1]:edge[1] + edge[3], edge[0]:edge[0] + edge[2]] = 1
    mask = RLSO(mask, rlso_h, rlso_w)
    candidate_components = cc.get_connected_components(mask)
    new_components = []
    for (idx, component) in enumerate(components):
        component_w = component[1].stop - component[1].start
        component_h = component[0].stop - component[0].start
        longer_side = max(component_w, component_h)
        shorter_side = min(component_w, component_h)
        if longer_side > max_allowed_long_side and shorter_side > max_allowed_short_side:
            new_components.append(component)
            continue
        add_list = [component]
        component_area = (component[0].stop - component[0].start) * (component[1].stop - component[1].start)
        y_slice = slice(max(component[0].start - expand_h, 0), min(component[0].stop + expand_h, img.shape[0]))
        x_slice = slice(max(component[1].start - expand_w, 0), min(component[1].stop + expand_w, img.shape[1]))
        new_slice = (y_slice, x_slice)
        for tmp_component in candidate_components:
            if overlap(new_slice, tmp_component):
                add_list.append(tmp_component)
        new_x_start = [tmp[1].start for tmp in add_list]
        new_y_start = [tmp[0].start for tmp in add_list]
        new_x_stop = [tmp[1].stop for tmp in add_list]
        new_y_sotp = [tmp[0].stop for tmp in add_list]
        new_component_area = (max(new_x_stop) - min(new_x_start)) * (max(new_y_sotp) - min(new_y_start))
        if new_component_area / component_area < 4:
            new_components.append((slice(min(new_y_start), max(new_y_sotp)), slice(min(new_x_start), max(new_x_stop))))
        else:
            new_components.append(component)
    return new_components


# '''
def analyze_block_vertical(mask, max_horizontal_text_height, space_width=1):
    """ Analyze whether a block is text-like by its segmentation sizes in the vertical direction

    Args:
        mask: masks resulting from the edge-based segmentation process
        max_horizontal_text_height: if the block height is less than this value, then consider it as a horizontal block


    Returns:
        True if the block looks good, and False otherwise
    """
    vertical_borders = find_border_vertical(mask)
    border_ok = border_analyze_vertical(mask, vertical_borders)
    if mask.shape[0] < max_horizontal_text_height and mask.shape[1] / mask.shape[0] > 5:
        return border_ok
    if not border_ok:
        return False
    widths = [abs(start - stop) for (start, stop) in vertical_borders]
    vertical_big_borders = []
    segmented_components = []
    for idx, w in enumerate(widths):
        if w >= space_width:
            vertical_big_borders.append(vertical_borders[idx])
    if len(vertical_big_borders) == 0:
        return False
    slice_start = 0
    vertical_slice_start_stops = []
    for v_border in vertical_big_borders[::-1]:
        vertical_slice_start_stops.append([slice_start, v_border[1]])
        # mask_slice_list.append(mask_slice)
        slice_start = v_border[0]
    vertical_slice_start_stops.append([slice_start, mask.shape[1] - 1])
    vertical_slice_start_stops = [(start, stop) for (start, stop) in vertical_slice_start_stops if stop > start]
    # print(vertical_slice_start_stops)
    for (vertical_start, vertical_stop) in vertical_slice_start_stops:
        mask_slice = mask[:, vertical_start:vertical_stop]
        # print('sub horizon borders')
        horizon_borders = find_border_horizon(mask_slice)
        # print(horizon_borders)
        if border_analyze_horizon(mask_slice, horizon_borders):
            return True
        # heights = [abs(border[0] - border[1]) for border in horizon_borders]
        # horizon_big_borders = [horizon_borders[idx] for (idx, h) in enumerate(heights) if h > space_height]
        # if len(horizon_big_borders) == 0:
        #     segmented_components.append([0, mask.shape[0] - 1, vertical_start, vertical_stop])
        # else:
        #     h_slice_start = 0
        #     for idx, h_border in enumerate(horizon_big_borders):
        #         if h_slice_start != h_border[0]:
        #             segmented_components.append([h_slice_start, h_border[0], vertical_start, vertical_stop])
        #         h_slice_start = h_border[1]
        #     if h_slice_start < mask.shape[0] - 1:
        #         segmented_components.append([h_slice_start, mask.shape[0] - 1, vertical_start, vertical_stop])
    return False


def analyze_block_horizon(mask):
    """ Analyze whether a block is text-like by its segmentation sizes in the horizontal direction

    Args:
        mask: masks resulting from the edge-based segmentation process
    Returns:
        True if the block looks good, and False otherwise
    """

    horizon_borders = find_border_horizon(mask)
    if border_analyze_horizon(mask, horizon_borders) and mask.shape[0] > mask.shape[1]:
        return True
    else:
        return False
    # heights = [abs(start - stop) for (start, stop) in horizon_borders]
    # horizon_big_borders = []
    # segmented_components = []
    # for idx, h in enumerate(heights):
    #     if h > space_height:
    #         horizon_big_borders.append(horizon_borders[idx])
    # if len(horizon_big_borders) == 0:
    #     return segmented_components
    # slice_start = 0
    # horizon_slice_start_stops = []
    # for h_border in horizon_big_borders[::-1]:
    #     horizon_slice_start_stops.append([slice_start, h_border[0]])
    #     slice_start = h_border[1]
    # horizon_slice_start_stops.append([slice_start, mask.shape[0] - 1])
    # horizon_slice_start_stops = [(start, stop) for (start, stop) in horizon_slice_start_stops if stop > start]
    # for (horizon_start, horizon_stop) in horizon_slice_start_stops:
    #     mask_slice = mask[horizon_start:horizon_stop, :]
    #     vertical_borders = find_border_vertical(mask_slice)
    #     widths = [abs(border[0] - border[1]) for border in vertical_borders]
    #     vertical_big_borders = [vertical_borders[idx] for (idx, w) in enumerate(widths) if w > space_width]
    #     if len(vertical_big_borders) == 0:
    #         segmented_components.append([horizon_start, horizon_stop, 0, mask.shape[1] - 1])
    #     else:
    #         v_slice_start = 0
    #         for idx, v_border in enumerate(vertical_big_borders[::-1]):
    #             if v_slice_start != v_border[1]:
    #                 segmented_components.append([horizon_start, horizon_stop, v_slice_start, v_border[1]])
    #             v_slice_start = v_border[0]
    #         if v_slice_start < mask.shape[1] - 1:
    #             segmented_components.append([horizon_start, horizon_stop, v_slice_start, mask.shape[1] - 1])
    # return segmented_components


# '''


def edge_cluster_contrast(img, edges):
    """ Functions to compute the contrasts of a cluster of edge boxes
    """
    contrasts = []
    for edge in edges:
        x, y, w, h = edge
        patch = img[y:y + h, x:x + w]
        contrasts.append(compute_contrast(patch))
    return contrasts


def find_border_vertical(mask):
    """ Functions to compute vertical borders that can segment an img into smaller segments
    """
    h, w = mask.shape
    border_start = h
    borders = []
    in_border = False
    for line in range(w - 1, -1, -1):
        if np.sum(mask[:, line]) == 0:
            if (not in_border):
                border_start = line
                in_border = True
        else:
            if in_border:
                borders.append([border_start, line])

            in_border = False
    if in_border: borders.append([border_start, 0])
    return borders


def find_border_horizon(mask):
    """ Functions to compute horizontal borders that can segment an img into smaller segments
    """
    h, w = mask.shape
    border_start = h
    borders = []
    in_border = False
    for line in range(0, h, 1):
        if np.sum(mask[line, :]) == 0:
            if (not in_border):
                border_start = line
                in_border = True
        else:
            if in_border:
                borders.append([border_start, line])

            in_border = False
    if in_border: borders.append([border_start, h - 1])
    return borders


def component_filter(components, img, edge_boxes, max_horizontal_txt_height=defaults.MAX_HORIZONTAL_TEXT_HEIGHT):
    """ Functions to filter out non-text boxes by segment widths/heights
    and white/black ratios in its neighboring areas

    Args:
        components: candiate boxes to be filtered through
        img: the img being processed
        edge_boxes: detected edges in the image
        max_side_length: the maximum length of box sides

    Returns:
        Text-like boxes
    """
    white_txt_background = False
    text_like_component = []
    num_of_box_with_white_neighbors = 0
    white_neighbors = []
    for component in components:
        mask = np.zeros(img.shape[0:2])
        edges = find_edges(component, edge_boxes)  # find edge areas inside the text box
        if len(edges) == 0: continue  # no processing if there is no edge in the box
        if max(edge_cluster_contrast(img, edges)) < 50: continue  # no processing if the contrast is too low
        adjusted_x, adjusted_y, w, h = edge_cluster_rectangle(
            edges)  # adjust the coordinates of the text box to make it tighter
        component = (slice(adjusted_y, adjusted_y + h), slice(adjusted_x, adjusted_x + w))
        ###### create a mask in which edge areas are filled with 1, other areas 0.
        for edge in edges:
            x, y, w, h = edge
            mask[y:y + h, x:x + w] = 1
        ############ crop the mask into the same shape as the box
        mask = mask[component[0].start:component[0].stop, component[1].start:component[1].stop]
        ############ extract the area of the text box from the image #################
        aoi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]
        aoi = clean.binarize(aoi, threshold=180)

        ############## compute the white/black ratio #######################
        zero_ratio = density_analysis(aoi)
        if zero_ratio > 0.75: continue  # if too many white pixels, drop it
        if zero_ratio < 0.15: continue  # if too many black pixels, drop it

        # print('--------------------------------------------------------')
        # analyze_block_vertical(mask)
        # ax1 = plt.subplot(1,2,1)
        # ax1.imshow(aoi)
        # ax2 = plt.subplot(1,2,2)
        # ax2.imshow(mask)
        # plt.show()

        ############### analyze the masks or aois to see whether it is text-like ##########

        if analyze_block_vertical(mask, max_horizontal_text_height=max_horizontal_txt_height) \
                or analyze_block_horizon(mask) \
                or analyze_block_vertical(aoi / 255, max_horizontal_text_height=max_horizontal_txt_height) \
                or analyze_block_horizon(aoi / 255):
            # if border_analyze_vertical(mask, vertical_borders) or border_analyze_horizon(mask, horizon_borders):
            text_like_component.append(component)
            ########## extract left, right, upper, lower neighboring areas of the candiate box ########
            component_left_neighbor = img[component[0].start:component[0].stop,
                                      max(component[1].start - 10, 0):component[1].start]
            component_right_neighbor = img[component[0].start:component[0].stop,
                                       component[1].stop:min(component[1].stop + 10, img.shape[1])]

            component_up_neighbor = img[max(component[0].start - 10, 0):component[0].start,
                                    component[1].start:component[1].stop]

            component_low_neighbor = img[component[0].stop:min(component[0].stop + 10, img.shape[0]),
                                     component[1].start:component[1].stop]
            ############# if  the candidate box is indeed a text box, it should should have white areas next to it #######
            left_white_ratio = 0
            if component_right_neighbor.shape[1] > 0 and component_right_neighbor.shape[0] > 0:
                left_white_ratio = np.sum(component_right_neighbor > 240) / (
                        component_right_neighbor.shape[0] * component_right_neighbor.shape[1])
            right_white_ratio = 0
            if component_left_neighbor.shape[0] > 0 and component_left_neighbor.shape[1] > 0:
                right_white_ratio = np.sum(component_left_neighbor > 240) / (
                        component_left_neighbor.shape[0] * component_left_neighbor.shape[1])
            up_white_ratio = 0
            if component_up_neighbor.shape[0] > 0 and component_up_neighbor.shape[1] > 0:
                up_white_ratio = np.sum(component_up_neighbor > 240) / (
                        component_up_neighbor.shape[0] * component_up_neighbor.shape[1])
            low_white_ratio = 0
            if component_low_neighbor.shape[0] > 0 and component_low_neighbor.shape[1] > 0:
                low_white_ratio = np.sum(component_low_neighbor > 240) / (
                        component_low_neighbor.shape[0] * component_low_neighbor.shape[1])
            white_neighbors.append(
                [left_white_ratio > 0.9, right_white_ratio > 0.9, up_white_ratio > 0.9, low_white_ratio > 0.9])
            if all([left_white_ratio > 0.95, right_white_ratio > 0.95, up_white_ratio > 0.95, low_white_ratio > 0.95]):
                num_of_box_with_white_neighbors = num_of_box_with_white_neighbors + 1

    if num_of_box_with_white_neighbors >= 2:  # if there are at least two boxes having neighbors all white, then all text areas have white background
        white_txt_background = True
        text_like_component = [component for idx, component in enumerate(text_like_component) if
                               np.sum(white_neighbors[idx]) >= 2]
    # text_like_component=expand_component_1(img,text_like_component,edge_boxes)
    return text_like_component, white_txt_background



def component_filter_by_color(components, img):
    """ Functions to filter out non-text boxes by checking whether their neighboring areas are white
    Args:
        components: candiate boxes to be filtered through
        img: the img being processed
        edge_boxes: detected edges in the image
        max_side_length: the maximum length of box sides

    Returns:
        Text-like boxes
    """
    new_component = []
    for component in components:
        component_left_neighbor = img[component[0].start:component[0].stop,
                                  max(component[1].start - 10, 0):component[1].start]
        component_right_neighbor = img[component[0].start:component[0].stop,
                                   component[1].stop:min(component[1].stop + 10, img.shape[1])]
        component_up_neighbor = img[max(component[0].start - 10, 0):component[0].start,
                                component[1].start:component[1].stop]
        component_low_neighbor = img[component[0].stop:min(component[0].stop + 10, img.shape[0]),
                                 component[1].start:component[1].stop]
        left_white_ratio = np.sum(component_right_neighbor > 240) / (
                component_right_neighbor.shape[0] * component_right_neighbor.shape[1])
        right_white_ratio = np.sum(component_left_neighbor > 240) / (
                component_left_neighbor.shape[0] * component_left_neighbor.shape[1])
        up_white_ratio = np.sum(component_up_neighbor > 240) / (
                component_up_neighbor.shape[0] * component_up_neighbor.shape[1])
        low_white_ratio = np.sum(component_low_neighbor > 240) / (
                component_low_neighbor.shape[0] * component_low_neighbor.shape[1])
        if np.sum([left_white_ratio > 0.9, right_white_ratio > 0.9, up_white_ratio > 0.9, low_white_ratio > 0.9]) > 2:
            new_component.append(component)
    return new_component
