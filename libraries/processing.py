import os
import glob
import cv2
import yaml
import numpy as np


def video_to_frames(url_vid: str) -> tuple:

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


# def clean_directories(BAD_BBOX, JSON_FRAMES, JSON_LABELS, VIDEO):

#     def clean_dir(relative_path, ext):

#         files = glob.glob(os.path.join(relative_path, '*' + ext))
#         for f in files:
#             os.remove(f)   

#     clean_dir(BAD_BBOX, 'png')
#     clean_dir(JSON_FRAMES, 'json')
#     clean_dir(JSON_LABELS, 'json')
#     clean_dir(VIDEO, 'mp4')

#     print('data directories emptied')
    

def get_action_image(action_id: str, LABELS: str) -> np.ndarray:

    '''
    get action id (like 1.1) and return image frame for label
    '''

    path_lb = os.path.join(LABELS , action_id + '.png')
    label = cv2.imread(path_lb, cv2.IMREAD_COLOR)

    return label



'''
functions for new version : whole video in one api call for 23 actions
'''
    

def convert_time_to_sec(str: str) -> int:

    '''
    str : format like '2114'
    return : 14 + 60*21 = 1274 second
    '''

    sec = int(str[2:])

    min = int(str[:2])

    return sec + min * 60


# def split_times_string(str: str) -> list:

#     '''
#     str : format like '001100580150.....2114'
#     return : list of time strings ['0011','0058',...,'2114']
#     '''

#     list = [str[i:i+4] for i in range(0, len(str), 4)]

#     return list



def get_times() -> list:

    '''
    get times from yaml file
    return list of times in seconds, ordered from 0s to max second
    '''

    outfile = open('data/times.yaml')
        
    dict = yaml.load(outfile, Loader=yaml. FullLoader)

    times = list(dict.values())

    result = []

    for time in times:
        
        sec = convert_time_to_sec(time)

        result.append(sec)

    return sorted(result)


def get_acions() -> list:

    '''
    get actions from yaml file
    return list of actions
    '''

    outfile = open('data/actions.yaml')
        
    dict = yaml.load(outfile, Loader=yaml. FullLoader)

    actions = list(dict.values())

    return actions



def video_to_frames_noFPS(url_vid: str) -> list:

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
            
            if i % round(fps) == 0:
                list_frames.append(image)
                k = k+1
            i = i+1
    except:
        pass

    return list_frames


def extract_frames(times: list, frames: list) -> list:

    '''
    extract frames at specific times in the list of action times
    return list of frames per ection

    times : list of times for each action, sorted, in seconds
    frames : list of frames extracted from the video
    fps : frame per second rate

    return list of frames containing actions
    '''

    result = []

    for i, sec in enumerate(times):

        frame = frames[sec]

        result.append(frame)

    return result

    
# print(convert_time_to_sec('0018'))

# print(split_times_string('001100580150'))

# print(get_times())

# print(get_acions())

# list_frames = video_to_frames_noFPS('data/test.mp4')

# list_times = get_times()

# actions = get_acions()

# res = extract_frames(list_times,list_frames)

# path = 'data/test_extraction'

# for i,r in enumerate(res):

#     p = os.path.join(path , actions[i] + '.png')

#     cv2.imwrite(p,r)


