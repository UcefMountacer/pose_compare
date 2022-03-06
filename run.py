
from libraries.processing import *
from libraries.score import *
from libraries.visualize import *
import subprocess

LABELS = 'labels/'
SAVE_DATA = 'data/frames/'
SAVE_LABEL = 'data/label/'
BAD_BBOX = 'outputs/bad_bbox/'
TH = 80.0


def run(action_id, video_path):


    # convert video to frames
    nbr_frames = video_to_frames(video_path, SAVE_DATA)

    # action image (label)
    # get_action_image(action_id, LABELS, SAVE_LABEL)   # pose estimate for all labels

    # run pose estimation for label and all frames
    # label first, then 
    # other frames
    # output json will contain all
    # to know how to get per frame json, use number of frames in the list


    command = 'python3 third_party/scripts/demo_inference.py \
                    --cfg third_party/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                    --checkpoint third_party/pretrained_models/fast_res50_256x192.pth \
                    --indir data/label/ \
                    --outdir outputs/json_label/'
               
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]


    command = 'python3 third_party/scripts/demo_inference.py \
                --cfg third_party/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                --checkpoint third_party/pretrained_models/fast_res50_256x192.pth \
                --indir data/frames/ \
                --outdir outputs/json/'

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

    # normalize
    label_json_normalized = l2_normalize('outputs/json/label.json')
    frames_json_normalized = l2_normalize('outputs/json/frames.json')

    # get action label (account for jpg and png types)
    action_label = [el for el in label_json_normalized if el['image_id'] == action_id + '.jpg'] + \
                   [el for el in label_json_normalized if el['image_id'] == action_id + '.png'] 

    # divide json into parts for each frame
    list_frames_data = divide_json_frames(frames_json_normalized, nbr_frames)

    # get median score for all frames and get max : this is the frame
    frame_data, frame_name = get_median_score_per_frame_and_max(list_frames_data , action_label)

    # get scores for this frame
    scores = cos_sim(action_label , frame_data)

    print(scores)

    #save bounding box of bad ppl pose (under 90)
    list_bad_bbox = bad_scores_box(frame_data, scores, TH)

    # save images of bbox
    save_bbox_img(list_bad_bbox , SAVE_DATA, frame_name, BAD_BBOX)
    













