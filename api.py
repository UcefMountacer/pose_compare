import os
from flask import (Flask, jsonify, request)
from flask_cors import CORS


app = Flask(__name__)

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



@app.route('/pose_compare')
def load_username():

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
    
    scores = run(action_id= action_Id , video_path= video_path)

    if scores == []:
        RESPONSE['status'] = False
        RESPONSE['message'] = 'video is empty'
        RESPONSE['data'] = {}
        return jsonify(RESPONSE)
    
    RESPONSE['status'] = True
    RESPONSE['message'] = 'pose comparison done Succesfully'
    RESPONSE['data']['scores'] = scores
    RESPONSE['data']['bad_poses'] = []

    return jsonify(RESPONSE)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

