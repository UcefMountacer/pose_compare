
import numpy as np
import json
import cv2

def read_json(json_path):

    with open(json_path) as f:
        file = json.load(f)

    print(len(file))
    return file


def cos_sim(label, ip_data):

    score_list = []
    final_scores = []

    for frame in range(len(ip_data)):
        ip_kpt = ip_data[frame]['keypoints']
        ip_list = []
        label_list = []

        for _ in range(17):
            temp_ip = []
            temp_la = []
            ip_list.append(ip_kpt[_ * 3])
            ip_list.append(ip_kpt[_ * 3 + 1])
            label_list.append(label[_ * 3])
            label_list.append(label[_ * 3 + 1])

            temp_ip.append(ip_kpt[_ * 3])
            temp_ip.append(ip_kpt[_ * 3 + 1])
            temp_la.append(label[_ * 3])
            temp_la.append(label[_ * 3 + 1])

            cs_temp = np.dot(temp_ip, temp_la) / \
                (np.linalg.norm(temp_ip)*np.linalg.norm(temp_la))
            score_list.append(((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100)

        cs_temp = np.dot(ip_list, label_list) / \
            (np.linalg.norm(ip_list)*np.linalg.norm(label_list))
        score = ((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100

        # print(score)
        final_scores.append(score)



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

    # with open(jsonfile.replace('.json', '_l2norm.json'), 'w') as f:
    #     json.dump(json_kps, f)
    #     print('Write l2_norm keypoints')

    return json_kps



def divide_json_frames(json_frames, nbr_frames):
    

    length = len(json_frames)
    n = length // nbr_frames

    list_frames_data = [json_frames[i:i + n] for i in range(0, len(json_frames), n)]

    return list_frames_data


def get_median_score_per_frame_and_max(list_frames_data, label_norm):

    list_sscores = []

    for frame_data in list_frames_data:

        scores = cos_sim(label_norm , frame_data)
        median = np.mean(scores)
        list_sscores.append(median)

    # max_score = np.max(LIST_SCORES)
    frame_data = list_frames_data[np.argmax(list_sscores)]
    index = np.argmax(list_sscores)

    return frame_data, index


def bad_scores_box(frame_data, scores):

    list_bad_bbox = []

    for sc, element in zip(scores, frame_data):

        if sc < 90:

            box = element[0]['box']
            list_bad_bbox.append(box)

    return list_bad_bbox


def save_bbox_img(list_bbox , path):

    #read image first
    im = cv2.imread(path)

