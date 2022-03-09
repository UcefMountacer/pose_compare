import os
from flask import (Flask, jsonify, request)
from flask_cors import CORS
import io
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
        'score': [],
        'bad_poses': []
    }
}


@app.route('/')
def home():

    RESPONSE['status'] = True
    RESPONSE['message'] = 'Nothing here, call the API'
    return jsonify(RESPONSE)



@app.route('/pose_compare' , methods=['GET' , 'POST'])
def pose_compare():

        action_Id = request.args.get('actionid')
        video_path = request.args.get('video_path')

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

        b = io.BytesIO()
        for i,im in enumerate(list_arrays):
            im = Image.fromarray(np.uint8(im*255))
            im.save(b, format="jpeg")
            b.seek(0)

        if scores == []:
            RESPONSE['status'] = False
            RESPONSE['message'] = 'video is empty'
            RESPONSE['data'] = {}
            return jsonify(RESPONSE)
        
        RESPONSE['status'] = True
        RESPONSE['message'] = 'pose comparison done Succesfully'
        RESPONSE['data']['scores'] = scores
        RESPONSE['data']['bad_poses'] = str(b.read())

        return jsonify(RESPONSE)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

