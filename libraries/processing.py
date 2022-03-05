import os
import cv2

def video_to_frames(path_vid, save_path):

    ''' 
    read video and return list of frames
    '''
    vidcap = cv2.VideoCapture(path_vid)
    i = 0
    success,image = vidcap.read()

    while success:
        success,image = vidcap.read()
        path = os.path.join(save_path , str(i)+ '.png')
        cv2.imwrite(path , image)
        i = i+1

    return i


        
def get_action_image(action_id, LABELS, path):

    '''
    get action id (like 1.1) and return image frame for label
    '''

    path = os.path.join(LABELS , action_id + '.png')
    label = cv2.imread(path)

    sav_path = os.path.join(path , action_id + '.png')
    cv2.imwrite(sav_path, label)

    