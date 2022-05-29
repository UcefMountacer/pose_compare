import os
import glob
import cv2

def video_to_frames(url_vid):

    ''' 
    read video and return list of frames
    '''

    vidcap = cv2.VideoCapture(url_vid)
    list_frames = []
    k = 0
    i = 0
    success,image = vidcap.read()
    try:
        while success:
            success,image = vidcap.read()
            
            if i%15 == 0:
                list_frames.append(image)
                k = k+1
            i = i+1
    except:
        pass

    return list_frames, k


def video_to_frames_noFPS(url_vid):

    ''' 
    read video and return list of frames
    '''

    vidcap = cv2.VideoCapture(url_vid)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    list_frames = []
    k = 0
    i = 0
    success,image = vidcap.read()
    try:
        while success:
            success,image = vidcap.read()
            
            list_frames.append(image)

    except:
        pass

    return list_frames, len(list_frames), fps


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

    path_lb = os.path.join(LABELS , action_id + '.jpg')
    label = cv2.imread(path_lb, cv2.IMREAD_COLOR)

    return label
    

def convert_time_to_sec(str):

    '''
    str : format like '2114'
    return : 14 + 60*21 = 1274 second
    '''

    sec = int(str[2:])

    min = int(str[:2])

    return sec + min * 60


def split_times_string(str):

    '''
    str : format like '001100580150.....2114'
    return : list of time strings ['0011','0058',...,'2114']
    '''

    list = [str[i:i+4] for i in range(0, len(str), 4)]

    return list

    
# print(convert_time_to_sec('0018'))

print(split_times_string('001100580150'))