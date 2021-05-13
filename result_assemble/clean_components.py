"""
Module:clean_components
Desc: beautify the results
Author: Wei Ou
DATE: April 7th 2020
"""

import numpy as np
from edge_detector.component_filter import overlap, find_edges, component_filter_by_color
from result_assemble.combine_box import merge_components, smaller_contained_by_larger


def DFS(box, neighbor_matrix, visited=[], result=[]):
    visited.append(box)
    neighbors = np.where(neighbor_matrix[box] == 1)[0].tolist()
    neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
    if len(neighbors) != 0:
        for neighbor in neighbors:
            DFS(neighbor, neighbor_matrix, visited, result)
    result.append(box)
    return


def finding_overlapped_components(box_list):
    neighbor_matrix = np.zeros([len(box_list), len(box_list)])
    for (idx, box) in enumerate(box_list):
        # box_center =[box_x+int(box_w/2),box_y+int(box_h/2) ]
        for next_idx in range(idx + 1, len(box_list)):
            if neighbor_matrix[idx, next_idx] != 0: continue
            next_box = box_list[next_idx]
            if overlap(box, next_box):
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


def delete_component_with_child(components):
    no_child_components = []
    for (idx1, tmp1) in enumerate(components):
        has_child = False
        for (idx2, tmp2) in enumerate(components):
            if idx1 == idx2:
                continue
            if smaller_contained_by_larger(tmp2, tmp1):
                has_child = True
                break
        if not has_child:
            no_child_components.append(tmp1)
    return no_child_components


def merge_overlapped(components):
    components = delete_component_with_child(components)
    component_clusters = finding_overlapped_components(components)
    overlappned_components = [members for members in component_clusters if len(members) > 1]
    isolated_components = [members[0] for members in component_clusters if len(members) == 1]
    isolated_components = [bx for (idx, bx) in enumerate(components) if idx in isolated_components]
    for tmp in overlappned_components:
        tmp = [bx for (idx, bx) in enumerate(components) if idx in tmp]
        smallest = sorted(tmp, key=lambda x: (x[0].stop - x[0].start) * (x[1].stop - x[1].start))[0]
        merged = merge_components(tmp)
        if smallest[0].stop - smallest[0].start > 100 or smallest[1].stop - smallest[1].start > 100:
            isolated_components = isolated_components + tmp
        else:
            isolated_components.append(merged)
    return isolated_components
