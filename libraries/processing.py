import os
import glob
import cv2

def video_to_frames(path_vid, save_path):

    ''' 
    read video and return list of frames
    '''

    vidcap = cv2.VideoCapture(path_vid)
    list_frames = []
    k = 0
    success,image = vidcap.read()
    try:
        while success:
            success,image = vidcap.read()
            
            if i%10 == 0:
                list_frames.append(image)
                k = k+1
            i = i+1
    except:
        pass

    return list_frames, k


def clean_directories(BAD_BBOX, JSON_FRAMES, JSON_LABELS, VIDEO):

    def clean_dir(relative_path, ext):

        files = glob.glob(os.path.join(relative_path, '*' + ext))
        for f in files:
            os.remove(f)   

    clean_dir(BAD_BBOX, 'png')
    clean_dir(JSON_FRAMES, 'json')
    clean_dir(JSON_LABELS, 'json')
    clean_dir(VIDEO, 'mp4')

    print('data directories emptied')
    

def get_action_image(action_id, LABELS):

    '''
    get action id (like 1.1) and return image frame for label
    '''

    path_lb = os.path.join(LABELS , action_id + '.png')
    label = cv2.imread(path_lb, cv2.IMREAD_COLOR)

    if label == None:
      path_lb = os.path.join(LABELS , action_id + '.jpg')
      label = cv2.imread(path_lb, cv2.IMREAD_COLOR)

    return label

    