

from flask import (Flask, jsonify, request)
from flask_cors import CORS
import base64
from third_party.OpenPose.net import *
from third_party.MtCnn.detector import *

app = Flask(__name__)

IMAGES_PATH = 'outputs/bad_bbox/'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

from run_v2 import *

RESPONSE = {
    'status': False,
    'message': '',
    'data': {
        'result': [],
        'bad_poses': []
    }
}

LIST_ACTION = get_acions()

@app.route('/')
def home():

    RESPONSE['status'] = True
    RESPONSE['message'] = 'call the API using url'
    RESPONSE['data'] = {}
    return jsonify(RESPONSE)



@app.route('/pose_compare')
def pose_compare():

        video_path = request.args.get('videoUrl')

        print('request received')

        if not video_path:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video path is required'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        
        net = load_model()
        det = init_detector()

        results = []

        list_times = get_times()

        list_frames = video_to_frames_noFPS(video_path)
        print('video to frames conversion done ...')

        action_frames = extract_frames(list_times,list_frames)
        print('action frames were extracted ...')

        for i, frame in enumerate(action_frames):

            action_id = LIST_ACTION[i]

            if action_id == '1.1-Smile':

                scores, bad_face = run_face_compare(det, action_id, frame)

                results.append([action_id,scores])

            else :

                scores, bad_pose = run_pose_compare(net, action_id, frame)

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

