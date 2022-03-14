import os
from flask import (Flask, jsonify, request)
from flask_cors import CORS
import io
import base64
from PIL import Image

app = Flask(__name__)

IMAGES_PATH = 'outputs/bad_bbox/'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

from run import *

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
        
        net = load_model()
        scores , list_arrays = run_pose_compare(net, action_Id, video_path)

        if scores == []:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video is empty'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)

        _, buffer = cv2.imencode('.png', list_arrays[0])
        s = base64.b64encode(buffer).decode("utf-8")
        
        RESPONSE['status'] = True
        RESPONSE['message'] = 'pose comparison done Succesfully'
        RESPONSE['data']['scores'] = scores
        RESPONSE['data']['bad_poses'] = s

        return jsonify(RESPONSE)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

