"""
Module: mser
Desc: detect text box by mser
Author: Wei Ou
DATE: April 7th 2020
"""

import cv2
import numpy as np
import scipy.ndimage
import skimage.morphology
import edge_detector.defaults as defaults
from skimage.feature import greycomatrix, greycoprops
from edge_detector import clean_page as clean, connected_components as cc
# from edge_detector.neighbors import finding_neighbors
from edge_detector.run_length_smoothing import RLSO
from result_assemble.combine_box import smaller_contained_by_larger
from edge_detector.component_filter import find_border_horizon, find_border_vertical, border_analyze_horizon, \
    border_analyze_vertical


def DFS(box, neighbor_matrix, visited=[], result=[]):
    """Depth-first-search function for box clustering based on their proximity
    """
    visited.append(box)
    neighbors = np.where(neighbor_matrix[box] == 1)[0].tolist()
    neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
    if len(neighbors) != 0:
        for neighbor in neighbors:
            DFS(neighbor, neighbor_matrix, visited, result)
    result.append(box)
    return


def adjacent(box1, box2, max_distance=defaults.ADJACENT_MAX_DISTANCE):
    """Function to check whether two boxes are adjacent in the coordinates
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1 - x2) < min(max_distance, w1 * 0.5):
        if y1 < y2:
            distance = abs(y1 + h1 - y2)
        else:
            distance = abs(y2 + h2 - y1)
        return distance < 0.5 * h1
    if abs(y1 - y2) < min(max_distance, h1 * 0.5):
        if x1 < x2:
            distance = abs(x1 + w1 - x2)
        else:
            distance = abs(x2 + w2 - x1)
        return distance < w1
    return False


def finding_neighbors(box_list):
    """Function to cluster a list of boxes by their proximity

    Args:
        box_list: the coordinates of a list of boxes


    Returns:
        Box clusters

    """
    neighbor_matrix = np.zeros([len(box_list), len(box_list)])
    for (idx, box) in enumerate(box_list):
        (box_x, box_y, box_w, box_h) = box
        box_area = box_w * box_h
        # long_side = max(box_w,)
        # box_center =[box_x+int(box_w/2),box_y+int(box_h/2) ]
        for next_idx in range(idx + 1, len(box_list)):
            if neighbor_matrix[idx, next_idx] != 0: continue
            next_box = box_list[next_idx]
            (next_box_x, next_box_y, next_box_w, next_box_h) = next_box
            # next_box_center = [next_box_x+int(next_box_w/2),next_box_y+int(next_box_h/2) ]
            # aligned =min(abs(next_box_x - box_x),abs(next_box_y-box_y))<aligned_threashold
            # near =max(abs(next_box_center[0] - box_center[0]), abs(next_box_center[1] - box_center[1]))<dis_threashold
            next_box_area = next_box_w * next_box_h
            bigger_one = max(box_area, next_box_area)
            smaller_one = min(box_area, next_box_area)

            if adjacent(box, next_box) and (bigger_one / smaller_one < 3):
                neighbor_matrix[idx, next_idx] = 1
                neighbor_matrix[next_idx, idx] = 1
            else:
                neighbor_matrix[idx, next_idx] = -1
                neighbor_matrix[next_idx, idx] = -1
    processed = []
    clusters = []
    for i in range(neighbor_matrix.shape[0]):
        if i not in processed:
            result = []
            DFS(i, neighbor_matrix, [], result)
            result = list(set(result))
            clusters.append(result)
            processed.extend(result)
    return clusters


def getEccentricity(mu):
    """Function to compute the eccentricty of a small area

    Args:
        mu: the momentum of the area


    Returns:
        The eccentricity of the area

    """
    bigSqrt = ((mu.m20 - mu.m02) * (mu.m20 - mu.m02) + 4 * mu.m11 * mu.m11) ** 0.5
    return (mu.m20 + mu.m02 + bigSqrt) / (mu.m20 + mu.m02 - bigSqrt)


def compute_contrast(aoi):
    """Function to compute the contrast of a small area
    Args:
        aoi: gray-scale aoi
    Returns:
    """
    glcm = greycomatrix(aoi, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return (greycoprops(glcm, 'dissimilarity')[0, 0])


def cluster_contrast(img, cluster):
    """Function to compute the contrast of a group of areas

    """
    contrasts = []
    for component in cluster:
        xs = component[1]
        ys = component[0]
        patch = img[ys.start:ys.stop, xs.start:xs.stop]
        contrasts.append(compute_contrast(patch))
    return contrasts


def find_child_components(big_component, small_components):
    """Find smaller compoentns located in a larger box

    """
    child_components = []
    for component in small_components:
        if smaller_contained_by_larger(component, big_component):
            child_components.append(component)
    return child_components


def stroke(regions, boxes, idx, img_size):
    cimg = np.zeros(img_size)
    region_points = regions[idx]
    [x, y, w, h] = boxes[idx]
    cimg[tuple(list(tuple(np.array(region_points).T))[::-1])] = 255
    cimg = cimg[y:y + h, x:x + w]
    cimg = np.pad(cimg, [1, 1], mode='constant', constant_values=0)
    not_cimg = cv2.bitwise_not(cimg)
    distanceImage = scipy.ndimage.morphology.distance_transform_edt(not_cimg)
    skeletonImage = skimage.morphology.thin(cimg, max_iter=None)
    strokeWidthValues = distanceImage[skeletonImage]
    return np.std(strokeWidthValues) / np.mean(strokeWidthValues)


def mser(gray, sigma=0.72):
    """ Detect text boxes by the mser technique
    Args:
        gray: gray-scale image
        sigma: parameter for the gaussian filter
    Returns:
        Text-like areas
    """
    vis = gray
    # vis = clean.grayscale(gray)
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(scipy.ndimage.gaussian_filter(vis, sigma))
    ######## drop regions that have 0 width or height ########
    bb_areas = [box[2] * box[3] for box in boxes if box[2] > 0 and box[3] > 0]
    ######### filter extremely small or big boxes ###################
    bb_area_median = np.median(bb_areas)
    max_bb_area = bb_area_median * 6
    min_bb_area = bb_area_median * 0.5
    regions = [region for idx, region in enumerate(regions) if
               bb_areas[idx] > min_bb_area and bb_areas[idx] < max_bb_area]
    ######### get the convex hull of the detected stable regions ###################
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    mask = np.zeros(vis.shape)
    bounding_boxes = []
    candidate_regions = []
    ######## filter out regions by extent, solidity, eccentricity, etc. ###########
    for i in range(len(hulls)):
        ctn = hulls[i]
        x, y, w, h = cv2.boundingRect(ctn)
        if x == 0: continue
        if y == 0: continue
        if x + w + 1 >= vis.shape[1]: continue
        if y + h + 1 >= vis.shape[0]: continue
        if w == 0: continue
        if h == 0: continue
        region_area = float(len(regions[i]))
        bb_area = w * h
        hull_area = cv2.contourArea(ctn)
        if region_area == 0: continue
        if bb_area == 0: continue
        if hull_area == 0: continue
        aspect_ratio = w / h
        extent = region_area / bb_area
        solidity = region_area / hull_area
        (_, _), (MA, ma), angle = cv2.fitEllipse(regions[i])
        a = ma / 2
        b = MA / 2
        eccentricity = (pow(a, 2) - pow(b, 2)) ** 0.5
        eccentricity = round(eccentricity / a, 2)
        if solidity > 0.3 and (extent > 0.2 and extent < 0.9) \
                and eccentricity < 0.999 and aspect_ratio > 0.3 and aspect_ratio < 5:
            mask[y:y + h, x:x + w] = 1
            bounding_boxes.append([x, y, w, h])
            candidate_regions.append(regions[i])
    ####### merge boxes that overlap each other ####################
    good_components = cc.get_connected_components(mask)
    mask = np.zeros(gray.shape, np.uint8)  # ,'B')
    for component in good_components:
        mask[component] = 1
    good_components = cc.get_connected_components(mask)
    good_components = [component for component in good_components if cc.area_bb(component) < max_bb_area]
    ############ cluster candidate boxes by their proximity in the coordinates ##########
    component_box_for_clustering = [(component[1].start, component[0].start,
                                     component[1].stop - component[1].start,
                                     component[0].stop - component[0].start) for component in good_components]
    component_id_clusters = finding_neighbors(component_box_for_clustering)
    component_id_clusters = sorted(component_id_clusters, key=len, reverse=True)
    component_clusters = []
    for id_cluster in component_id_clusters:
        component_clusters.append([good_components[component_id] for component_id in id_cluster])
    ################## compute the contrasts of the region clusters #################
    component_cluster_contrasts = [max(cluster_contrast(gray, cluster)) for cluster in component_clusters]
    ################ keep the clusters whose contrast level exceeds the threshold ###############
    good_cluster = [cluster for (idx, cluster) in enumerate(component_clusters) if
                    component_cluster_contrasts[idx] >= 70 and len(cluster) > 1]
    good_components = [area for cluster in good_cluster for area in cluster]
    good_mask = np.zeros(gray.shape, np.uint8)  # ,'B')
    for area in good_components:
        good_mask[area] = 1
    good_mask = RLSO(good_mask, 50, 50)
    candidate_compoentns = cc.get_connected_components(good_mask)
    final_components = []
    #########
    for candiate in candidate_compoentns:
        child_components = find_child_components(candiate, good_components)
        mask = np.zeros(gray.shape[0:2])
        for child in child_components:
            mask[child] = 1
        mask = mask[candiate[0].start:candiate[0].stop, candiate[1].start:candiate[1].stop]
        vertical_borders = find_border_vertical(mask)
        horizon_borders = find_border_horizon(mask)
        if border_analyze_vertical(mask, vertical_borders) and border_analyze_horizon(mask, horizon_borders):
            final_components.append(candiate)
    return final_components
