
from libraries.processing import *
from libraries.score import *
from libraries.visualize import *
from third_party.inference import *

CWD = os.getcwd()
LABELS         = os.path.join(CWD ,'labels/')
BAD_BBOX       = os.path.join(CWD ,'outputs/bad_bbox/')
JSON_FRAMES    = os.path.join(CWD ,'outputs/json_frames/')
JSON_LABELS    = os.path.join(CWD ,'outputs/json_labels/')
# VIDEO          = os.path.join(CWD ,'data/')
TH = 80.0


def run_pose_compare(net, action_id, video_url):

    # first clean directories
    # clean_directories(BAD_BBOX, JSON_FRAMES, JSON_LABELS, VIDEO)

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

        # run for images
        all_res = run_demo(net,list_frames)
        print('pose done for video')

        # run for images
        res_label = run_demo(net,[label])
        print('poses done for label')

        # get median score for all frames and get max : this is the frame
        frame_data, frame_index = get_median_score_per_frame_and_max(all_res , res_label)

        # get scores for this frame
        scores = cos_sim(res_label , frame_data)

        #save bounding box of bad ppl pose (under 90)
        list_bad_bbox = bad_scores_box(frame_data, scores, TH)

        # save images of bbox
        list_arrays = save_bbox_img(list_bad_bbox , list_frames[frame_index], BAD_BBOX)

        print('bad poses saved, scores calculated from left to right')

        return scores, list_arrays



# net = load_model()

# scores = run_pose_compare(net, '3.1', 'test.mp4')
    













