
from libraries.processing import *
# from libraries.score import *
from libraries.visualize import *
from libraries.openpose import *
from libraries.face_analysis import *

CWD = os.getcwd()
LABELS = os.path.join(CWD ,'labels/')


def run_pose_compare(net, action_id, video_url):

    print('starting pose estimation...')

    # convert video to frames
    list_frames, nbr_frames = video_to_frames(video_url)
    print('video to frames conversion done')

    # get action label
    label = get_action_image(action_id, LABELS)
    print('label found successfully')

    if nbr_frames == 0:

        print('video empty')
        return []
    
    if nbr_frames > 0:

        #take last 5 images only
        list_frames = list_frames[-5:]

        # run for images
        all_res = run_posenet(net,list_frames)
        print('pose done for video')

        # run for images
        res_label = run_posenet(net,[label])
        print('poses done for label')

        # get median score for all frames and get max : this is the frame
        scores, frame_data, frame_index = get_pose_score(all_res , res_label)

        #save bounding box of bad ppl pose (under 90)
        bad_pose = bad_scores_box(frame_data, scores, list_frames[frame_index])

        print('bad poses saved, scores calculated from left to right')

        return scores, bad_pose

        

def run_face_compare(det, action_id, video_url):

    print('starting face estimation...')

    # convert video to frames
    list_frames, nbr_frames = video_to_frames(video_url)
    print('video to frames conversion done')

    # get action label
    label = get_action_image(action_id, LABELS)
    print('label found successfully')

    if nbr_frames == 0:

        print('video empty')
        return []
    
    if nbr_frames > 0:

        #take last 5 images only
        list_frames = list_frames[-5:]

        # run for images
        all_kpts, all_boxes = run_mtcnn(det, list_frames)
        print('faces done for video')

        # run for images
        label_kpts, _ = run_mtcnn(det, [label])
        print('face done for label')

        # get median score for all frames and get max : this is the frame
        scores, frame_boxes, frame_index = get_face_score(all_kpts, label_kpts, all_boxes)

        #save bounding box of bad ppl pose (under 90)
        bad_face = face_bad_scores_box(frame_boxes, scores, list_frames[frame_index])

        print('bad face saved, scores calculated from left to right')

        return scores, bad_face



# net = load_model()

# url = 'data/test.mp4'

# scores = run_pose_compare(net, '3.1', url)
    













