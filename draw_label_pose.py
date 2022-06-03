

import cv2
import numpy as np
import torch
import os

from third_party.OpenPose.models.with_mobilenet import PoseEstimationWithMobileNet
from third_party.OpenPose.modules.keypoints import extract_keypoints, group_keypoints
from third_party.OpenPose.modules.load_state import load_state
from third_party.OpenPose.modules.pose import Pose, track_poses
from third_party.OpenPose.val import normalize, pad_width
from third_party.OpenPose.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
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


def run_for_label(img: np.ndarray) -> np.ndarray:

    stride = 8
    upsample_ratio = 4
    num_keypoints = 18
    previous_poses = []
    delay = 1


    heatmaps, pafs, scale, pad = infer_fast(net, img, 256, stride, upsample_ratio, 1)
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

    white = np.full((img.shape[0], img.shape[1],3),255).astype(np.uint8)

    for pose in current_poses:
        pose.draw(white)

    return white


net = PoseEstimationWithMobileNet()
checkpoint = torch.load('third_party/OpenPose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)

net = net.eval()


if __name__ == "__main__":

    labels_path = 'labels/originals'
    res_path = 'labels/results'

    labels = os.listdir(labels_path)

    for l in labels:

        print(l)

        img_path = os.path.join(labels_path , l)
        img = cv2.imread(img_path)

        res = run_for_label(img)
        path = os.path.join(res_path , l)

        cv2.imwrite(path , res)

