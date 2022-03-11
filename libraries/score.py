
import numpy as np
import json
import os
import cv2

def read_json(json_path):

    with open(json_path) as f:
        file = json.load(f)

    print(len(file))
    return file


def cos_sim(res_label, res):

    '''
    take list of poses in image (res) and poses in label (res_label)
    return score
    '''

    score_list = []
    final_scores = []

    label = res_label[0][0]['keypoints']

    for pose in res:
        ip_kpt = pose['keypoints']
        ip_list = []
        label_list = []

        for kpoint, kpoint_label in zip(ip_kpt,label):
            temp_ip = []
            temp_la = []
            ip_list.append(kpoint[0])
            ip_list.append(kpoint[1])
            label_list.append(kpoint_label[0])
            label_list.append(kpoint_label[1])

            temp_ip.append(kpoint[0])
            temp_ip.append(kpoint[1])
            temp_la.append(kpoint_label[0])
            temp_la.append(kpoint_label[1])

            cs_temp = np.dot(temp_ip, temp_la) / \
                (np.linalg.norm(temp_ip)*np.linalg.norm(temp_la))
            score_list.append(((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100)

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


def l2_normalize(jsonfile):

    '''
    input json output from model and normalize it
    return json data
    '''

    with open(jsonfile) as kps:
        json_kps = json.load(kps)

        for frame in range(len(json_kps)):
            keypoints = json_kps[frame]['keypoints']
            box = json_kps[frame]['box']
            temp_x = np.abs(box[0] - box[2]) / 2
            temp_y = np.abs(box[1] - box[3]) / 2

            if temp_x <= temp_y:
                if box[0] <= box[2]:
                    sub_x = box[0] - (temp_y - temp_x)
                else:
                    sub_x = box[2] - (temp_y - temp_x)

                if box[1] <= box[3]:
                    sub_y = box[1]
                else:
                    sub_y = box[3]
            else:
                if box[1] <= box[3]:
                    sub_y = box[1] - (temp_x - temp_y)
                else:
                    sub_y = box[3] - (temp_x - temp_y)

                if box[0] <= box[2]:
                    sub_x = box[0]
                else:
                    sub_x = box[2]

            temp = []
            for _ in range(17):
                temp.append(keypoints[_ * 3] - sub_x)
                temp.append(keypoints[_ * 3 + 1] - sub_y)

            norm = np.linalg.norm(temp)
            for _ in range(17):
                keypoints[_ * 3] = (keypoints[_ * 3] - sub_x) / norm
                keypoints[_ * 3 + 1] = (keypoints[_ * 3 + 1] - sub_y) / norm
                json_kps[frame]['keypoints'] = keypoints

    return json_kps



# def divide_json_frames(json_frames, nbr_frames):
    

#     length = len(json_frames)
#     n = length // nbr_frames

#     list_frames_data = [json_frames[i:i + n] for i in range(0, len(json_frames), n)]

#     return list_frames_data


def get_median_score_per_frame_and_max(all_res, res_label):

    list_sscores = []

    for res in all_res:

        scores = cos_sim(res_label , res)
        median = np.mean(scores)
        list_sscores.append(median)

    # max_score = np.max(LIST_SCORES)
    frame_data = all_res[np.argmax(list_sscores)]
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
