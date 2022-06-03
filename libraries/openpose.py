
import cv2
import numpy as np
import torch

from third_party.OpenPose.models.with_mobilenet import PoseEstimationWithMobileNet
from third_party.OpenPose.modules.keypoints import extract_keypoints, group_keypoints
from third_party.OpenPose.modules.load_state import load_state
from third_party.OpenPose.modules.pose import Pose
from third_party.OpenPose.val import normalize, pad_width


def infer_fast(net, img, net_input_height_size=256, stride=8, upsample_ratio=4, cpu=1,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    
    '''
    run inference and all pre/post processing step on OpenPose
    return element specific to pose estimation (heatmap, ....)
    '''
    
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad




def frame_detection(net, frame, i):

    '''
    running net over one frame
    return a dictionnary containg bbox and keypoints and other infos
    
    '''

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    print('inferencing for frame ', str(i))

    heatmaps, pafs, scale, pad = infer_fast(net, frame)
    
    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    print('post processing pose data...')

    res = []
    for pose in current_poses:
        d = {}
        d['box']        = [(pose.bbox[0], pose.bbox[1]), (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3])]
        d['confidence'] = pose.confidence
        d['keypoints']  = pose.keypoints
        d['yc']  = (d['box'][1][1] - d['box'][0][1])/2 + d['box'][0][1]
        d['xc'] = (d['box'][1][0] - d['box'][0][0])/2 + d['box'][0][0]
        d['image_index'] = i
        res.append(d) 

    return res




def run_posenet(net, images_list):

    '''
    run posenet on a list of frames
    return a list of dictionnaries
    '''

    net = net.eval()
    
    # stride = 8
    # upsample_ratio = 4
    # num_keypoints = Pose.num_kpts
    
    All_res = []

    for i, img in enumerate(images_list):

        # run inference

        try:
            res = frame_detection(net, img, i)
        except:
            print('issue with a frame')
            continue

        # normalize

        res = normalize_bbox_kpts_pose(res)
    
        All_res.append(res)

    return All_res




def normalize_bbox_kpts_pose(res):

    '''
    normalizing bbox to 0-1 in width and height to ensure comparison 
    of dace keypoint is done in the same size
    '''
    for pose in res:

        kpt = pose['keypoints']

        [(x,y),(x2,y2)] = pose['box']
        dx = x2-x
        dy = y2-y

        norm_kpt = np.zeros(kpt.shape)
        for i,k in enumerate(kpt):

            a,b = k[0],k[1]

            if a != -1 or b != -1:

                na = (a-x)/dx
                nb = (b-y)/dy
                norm_kpt[i] = (na,nb)

            else:

                norm_kpt[i] = (0,0)

        pose['keypoints'] = norm_kpt

    return res


def score_pose(res_label, res):

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
    '''
    find centers of face boxes in each frame
    '''
  
    l = []
    for i,p in enumerate(res):
        yc = p['yc']
        xc = p['xc']
        l.append([i,xc,yc])
    
    return l




def get_pose_score(all_res, res_label):

    '''
    all_res : all pose dictionnaries of all frames
    res_label : pose dict of label
    '''

    list_scores = []

    for res in all_res:

        scores = score_pose(res_label , res)
        median = np.mean(scores)
        list_scores.append(median)

    # max_score = np.max(LIST_SCORES)
    frame_data = all_res[np.argmax(list_scores[2:])]
    frame_index = frame_data[0]['image_index']

    scores  = score_pose(res_label , frame_data)

    return scores, frame_data, frame_index


def get_pose_score_action30(res, res_label):

    '''
    res : pose dictionnaries of all persons in one frame
    res_label : pose dict of label
    '''

    scores  = score_pose(res_label , res[0])

    return scores


def bad_scores_box(frame_data, scores, frame_array):


    box = frame_data[np.argmin(scores)]['box']

    [(a,b),(c,d)] = box
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)

    bim = frame_array[b:d,a:c]

    return bim

 