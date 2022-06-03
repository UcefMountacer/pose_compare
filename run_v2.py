
from libraries.processing import *
# from libraries.score import *
from libraries.visualize import *
from libraries.openpose import *
from libraries.face_analysis import *
from third_party.OpenPose.net import *
import base64
import json

CWD = os.getcwd()
LABELS = os.path.join(CWD ,'labels/originals')


'''
FPS can be known from video cap
we have number of frames
so we can deduce duration of the video

480/24 = 20 sec video (this is not necessary)

if action 1 at 0001, so at 1 sec, then we take first 24 frames block (and choose last one)

if action x at 0010, so at 10 sec, we take the 10th 24 frames block, and choose last one

'''


def run_pose_compare_v2(net, action_id, frame):

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
    scores = get_pose_score_action30(res , res_label)

    #save bounding box of bad ppl pose (under 90)
    bad_pose = bad_scores_box(res[0], scores, frame)

    print('bad poses saved, scores calculated from left to right')

    return scores, bad_pose

        

def run_face_compare_v2(det, action_id, frame):

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
    scores, frame_boxes = get_face_score(kpts, label_kpts, boxes)

    #save bounding box of bad ppl pose (under 90)
    bad_face = face_bad_scores_box(frame_boxes, scores, frame)

    print('bad face saved, scores calculated from left to right')

    return scores, bad_face





# RESPONSE = {
#     'status': False,
#     'message': '',
#     'data': []
# }


# net = load_model()
    
# list_frames = video_to_frames_noFPS('data/test.mp4')

# list_times = get_times()

# LIST_ACTION = get_acions()

# action_frames = extract_frames(list_times,list_frames)

# results = []

# for i, frame in enumerate(action_frames):

#     action_id = LIST_ACTION[i]

#     scores, bad_pose = run_pose_compare(net, action_id, frame)

#     _, buffer = cv2.imencode('.png', bad_pose)
#     s = base64.b64encode(buffer).decode("utf-8")

#     results.append([action_id, scores, s])

# RESPONSE['status'] = True
# RESPONSE['message'] = 'pose comparison done Succesfully'
# RESPONSE['data'] = results


# with open('data.json', 'w') as fp:
#     json.dump(RESPONSE, fp)












