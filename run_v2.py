
from libraries.processing import *
# from libraries.score import *
from libraries.visualize import *
from libraries.openpose import *
from libraries.face_analysis import *
from third_party.OpenPose.net import *


CWD = os.getcwd()
LABELS = os.path.join(CWD ,'labels/')


'''
FPS can be known from video cap
we have number of frames
so we can deduce duration of the video

480/24 = 20 sec video (this is not necessary)

if action 1 at 0001, so at 1 sec, then we take first 24 frames block (and choose last one)

if action x at 0010, so at 10 sec, we take the 10th 24 frames block, and choose last one

'''


def run_pose_compare(net, action_id, frame):

    print('starting pose estimation...')

    # get action label
    label = get_action_image(action_id, LABELS)
    print('label found successfully')

    # run for image
    res = run_posenet(net,[frame])
    print('pose done for video frame')

    # run for label
    res_label = run_posenet(net,[label])
    print('poses done for label')

    # get median score for all frames and get max : this is the frame
    scores, frame_data, frame_index = get_pose_score(res , res_label)

    #save bounding box of bad ppl pose (under 90)
    # bad_pose = bad_scores_box(frame_data, scores, list_frames[frame_index])

    print('bad poses saved, scores calculated from left to right')

    return scores

        

def run_face_compare(det, action_id, frame):

    print('starting face estimation...')

    # get action label
    label = get_action_image(action_id, LABELS)
    print('label found successfully')

    # run for images
    kpts, boxes = run_mtcnn(det, [frame])
    print('faces done for video frame')

    # run for images
    label_kpts, _ = run_mtcnn(det, [label])
    print('face done for label')

    # get median score for all frames and get max : this is the frame
    scores, frame_boxes, frame_index = get_face_score(kpts, label_kpts, boxes)

    #save bounding box of bad ppl pose (under 90)
    # bad_face = face_bad_scores_box(frame_boxes, scores, list_frames[frame_index])

    print('bad face saved, scores calculated from left to right')

    return scores



# net = load_model()

# url = 'data/test.mp4'

# scores = run_pose_compare(net, '3.1', url)
    













