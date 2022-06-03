
import base64
from flask import (Flask, jsonify, request)
from flask_cors import CORS
from run import *
from run_v2 import *
from third_party.OpenPose.net import *
from third_party.MtCnn.detector import *

app = Flask(__name__)

IMAGES_PATH = 'outputs/bad_bbox/'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)


'''
singular actions 29
'''

RESPONSE = {
    'status': False,
    'message': '',
    'data': {
        'scores': [],
        'bad_poses': []
    }
}

@app.route('/')
def home():

    RESPONSE['status'] = True
    RESPONSE['message'] = 'call the API using url'
    RESPONSE['data'] = {}
    return jsonify(RESPONSE)


@app.route('/pose_compare')
def pose_compare():

        action_Id = request.args.get('actionId')
        video_path = request.args.get('videoUrl')

        print('request received')

        if not action_Id:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'action Id is required'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        if not video_path:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video is required'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        if action_Id != '1.1_v2':
        
            net = load_model()
            scores , bad_pose = run_pose_compare(net, action_Id, video_path)

        if action_Id == '1.1_v2':

            det = init_detector()
            scores , bad_pose = run_face_compare(det, action_Id, video_path)


        if scores == []:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video is empty'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        _, buffer = cv2.imencode('.png', bad_pose)
        s = base64.b64encode(buffer).decode("utf-8")
        
        RESPONSE['status'] = True
        RESPONSE['message'] = 'pose comparison done Succesfully'
        RESPONSE['data']['scores'] = scores
        RESPONSE['data']['bad_poses'] = s

        return jsonify(RESPONSE)


'''
action 30
'''

RESPONSE_action30 = {
    'status': False,
    'message': '',
    'data': []
}



LIST_ACTION = get_acions()
LIST_TIMES = get_times()

net = load_model()
det = init_detector()



@app.route('/pose_compare_action30')
def pose_compare_action30():

        video_path = request.args.get('videoUrl')

        print('request received')

        if not video_path:
            RESPONSE_action30['status'] = False
            RESPONSE_action30['message'] = 'video path is required'
            RESPONSE_action30['data'] = []
            return jsonify(RESPONSE_action30)

        results = []


        list_frames = video_to_frames_noFPS(video_path)
        print('video to frames conversion done ...')

        action_frames = extract_frames(LIST_TIMES,list_frames)
        print('action frames were extracted ...')

        for i, frame in enumerate(action_frames):

            action_id = LIST_ACTION[i]

            if action_id == '1.1-Smile':

                scores, bad_face = run_face_compare_v2(det, action_id, frame)
                _, buffer = cv2.imencode('.png', bad_face)
                
                s = base64.b64encode(buffer).decode("utf-8")

                results.append([action_id, scores, s])

            else :

                scores, bad_pose = run_pose_compare_v2(net, action_id, frame)
                _, buffer = cv2.imencode('.png', bad_pose)
                
                s = base64.b64encode(buffer).decode("utf-8")

                results.append([action_id, scores, s])


        if results == []:
            
            RESPONSE_action30['status'] = False
            RESPONSE_action30['message'] = 'video is empty'
            RESPONSE_action30['data'] = []
            return jsonify(RESPONSE_action30)
        
        RESPONSE_action30['status'] = True
        RESPONSE_action30['message'] = 'pose comparison done Succesfully'
        RESPONSE_action30['data'] = results

        return jsonify(RESPONSE_action30)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)

