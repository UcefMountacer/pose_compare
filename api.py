

from flask import (Flask, jsonify, request)
from flask_cors import CORS
import base64
from third_party.OpenPose.net import *
from third_party.MtCnn.detector import *

app = Flask(__name__)

IMAGES_PATH = 'outputs/bad_bbox/'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

from run import *

RESPONSE = {
    'status': False,
    'message': '',
    'data': {
        'result': [],
        'bad_poses': []
    }
}

LIST_ACTION = ['1.1,.......,']


@app.route('/')
def home():

    RESPONSE['status'] = True
    RESPONSE['message'] = 'call the API using url'
    RESPONSE['data'] = {}
    return jsonify(RESPONSE)



@app.route('/pose_compare')
def pose_compare():

        # action_times = request.args.get('actionTimes')

        video_path = request.args.get('videoUrl')

        print('request received')

        # if not action_times:
        #     RESPONSE['status'] = False
        #     RESPONSE['message'] = 'action times frame is required'
        #     RESPONSE['data'] = {}
        #     return jsonify(RESPONSE)

        if not video_path:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video path is required'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        
        net = load_model()
        det = init_detector()

        # scores , bad_pose = run_pose_compare(net, action_Id, video_path)
        # scores , bad_pose = run_face_compare(det, action_Id, video_path)

        results = []

        # list_times = split_times_string(action_times)

        list_times = get_times()

        list_frames, nbr_frames, fps = video_to_frames_noFPS(video_path)
        print('video to frames conversion done')

        for time, i in enumerate(list_times):

            action_id = LIST_ACTION[i]

            sec = convert_time_to_sec(time)

            frame_index = (sec+1) * fps

            frame = list_frames[frame_index-1]

            if action_id == '1.1':

                scores = run_face_compare(det, action_id, frame)

            else :

                scores = run_pose_compare(net, action_id, frame)

            bad_pose = 'TO DO'

            results.append([action_id,scores])


        if results == []:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video is empty'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        _, buffer = cv2.imencode('.png', bad_pose)
        s = base64.b64encode(buffer).decode("utf-8")
        
        RESPONSE['status'] = True
        RESPONSE['message'] = 'pose comparison done Succesfully'
        RESPONSE['data']['result'] = results
        RESPONSE['data']['bad_poses'] = s

        return jsonify(RESPONSE)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)

