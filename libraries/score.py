
import numpy as np
import json
import os
import cv2
import math

def cos_sim(res_label, res):

    '''
    take list of poses in image (res) and poses in label (res_label)
    return score
    '''

    final_scores = []

    label = res_label[0][0]['keypoints']

    for pose in res:
        ip_kpt = pose['keypoints']
        ip_list = []
        label_list = []

        for kpoint, kpoint_label in zip(ip_kpt,label):

            ip_list.append(kpoint[0])
            ip_list.append(kpoint[1])
            label_list.append(kpoint_label[0])
            label_list.append(kpoint_label[1])

        cs_temp = np.dot(ip_list, label_list) / \
            (np.linalg.norm(ip_list)*np.linalg.norm(label_list))
        score = ((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100

        final_scores.append(score)

    # sort poses from left to right

    l = find_centers(res)
    l.sort(key=lambda x:x[1])
    ind = [t[0] for t in l]
    ordered_scores = [final_scores[i] for i in ind]

    return ordered_scores


def find_centers(res):
  
    l = []
    for i,p in enumerate(res):
        yc = p['yc']
        xc = p['xc']
        l.append([i,xc,yc])
    
    return l




def get_pose_score(all_res, res_label):

    list_sscores = []

    for res in all_res:

        scores = cos_sim(res_label , res)
        median = np.mean(scores)
        list_sscores.append(median)

    # max_score = np.max(LIST_SCORES)
    list_sscores = [0 if math.isnan(x) else x for x in list_sscores]
    frame_data = all_res[np.argmax(list_sscores[2:])]
    frame_index = frame_data[0]['image_index']

    return frame_data, frame_index


def bad_scores_box(frame_data, scores, TH):

    # list_bad_bbox = []

    # for sc, element in zip(scores, frame_data):

    #     if sc < TH:

    #         box = element['box']
    #         list_bad_bbox.append(box)

    # return list_bad_bbox

    box = frame_data[np.argmin(scores)]['box']

    return [box]

    

def save_bbox_img(list_bbox , frame_array, sav_path):


    list_arrays = []

    #get bbox
    for i,box in enumerate(list_bbox):
        [(a,b),(c,d)] = box
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)

        bim = frame_array[b:d,a:c]
        # cv2.imwrite(os.path.join(sav_path, 'box_' + str(i) + '.png') , bim)
        list_arrays.append(bim)

    return list_arrays
