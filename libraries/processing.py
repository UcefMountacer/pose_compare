import os
import glob
import cv2

def video_to_frames(path_vid, save_path):

    ''' 
    read video and return list of frames
    '''

    vidcap = cv2.VideoCapture(path_vid)
    i = 0
    success,image = vidcap.read()
    try:
        while success:
            if i%10 == 0:
                success,image = vidcap.read()
                path = os.path.join(save_path , str(i)+ '.png')
                cv2.imwrite(path , image)
            i = i+1
    except:
        pass

    return i


def clean_directories(CWD, SAVE_DATA, BAD_BBOX, JSON_FRAMES, JSON_LABELS, video_path):

    def clean_dir(CWD, relative_path, ext):

        files = glob.glob(os.path.join(CWD , relative_path, '*' + ext))
        for f in files:
            os.remove(f)   

    clean_dir(CWD, SAVE_DATA, 'png')
    clean_dir(CWD, BAD_BBOX, 'png')
    clean_dir(CWD, JSON_FRAMES, 'json')
    clean_dir(CWD, JSON_LABELS, 'json')
    clean_dir(CWD, video_path, 'mp4')

    print('data directories emptied')
    




# # Unused

# def get_action_image(action_id, LABELS, path):

#     '''
#     get action id (like 1.1) and return image frame for label
#     '''

#     path_lb = os.path.join(LABELS , action_id + '.png')
#     label = cv2.imread(path_lb)

#     if label == None:
#       path_lb = os.path.join(LABELS , action_id + '.jpg')
#       label = cv2.imread(path_lb)

#     sav_path = os.path.join(path , action_id + '.png')
#     cv2.imwrite(sav_path, label)

    