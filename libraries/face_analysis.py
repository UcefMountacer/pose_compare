
import numpy as np



def run_detection(det, frame):

    '''
    det : detector instance
    frame: frame to infer using det
    '''

    faces = det.detect_faces(frame)
    frame_boxes = []
    frame_kpts = []

    for face in faces:
        # boxes
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        frame_boxes.append((x,y,x2,y2))

        # keypoints
        frame_kpts.append(face['keypoints'])

    return frame_boxes, frame_kpts


def normalize_bbox_kpts(frame_boxes, frame_kpts):

    '''
    normalizing bbox to 0-1 in width and height to ensure comparison 
    of dace keypoint is done in the same size
    '''

    frame_normalized_kpts = []

    for box, kpt in zip(frame_boxes,frame_kpts):

        x,y,x2,y2 = box
        dx = x2-x
        dy = y2-y
        
        norm_kpt = {}
        for item,value in kpt.items():

            a,b = value
            na = (a-x)/dx
            nb = (b-y)/dy

            # {'left_eye': (127, 62),
            # 'mouth_left': (127, 80),
            # 'mouth_right': (144, 81),
            # 'nose': (134, 74),
            # 'right_eye': (145, 63)}
            norm_kpt[item] = (na,nb)

        frame_normalized_kpts.append(norm_kpt)

    return frame_normalized_kpts, frame_boxes


def score_face(normalized_label_kpts, frame_normalized_kpts, frame_boxes):

    '''
    All_norm_kpts_label : contains one element for the label
    All_norm_kpts : contains all faces data from frame analysed
    Boxes : of frame not label
    '''

    final_scores = []

    label = normalized_label_kpts[0][0]
    
    for face in frame_normalized_kpts:


        ip_list = []
        label_list = []

        for (_,kpoint) , (_,kpoint_label) in zip(face.items(), label.items()):
       
            ip_list.append(kpoint[0])
            ip_list.append(kpoint[1])
            label_list.append(kpoint_label[0])
            label_list.append(kpoint_label[1])

        cs_temp = np.dot(ip_list, label_list) / \
            (np.linalg.norm(ip_list)*np.linalg.norm(label_list))
        score = ((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100

        final_scores.append(score)

    # return final_scores

    # sort poses from left to right

    l = find_centers_face(frame_boxes)
    l.sort(key=lambda x:x[1])
    ind = [t[0] for t in l]
    ordered_scores = [final_scores[i] for i in ind]

    return ordered_scores



def find_centers_face(frame_boxes):
    '''
    find centers of face boxes in each frame
    '''
  
    l = []
    for i,box in enumerate(frame_boxes):

        x,y,x2,y2 = box
        xc = (x2-x)/2 + x
        yc = (y2-y)/2 + y

        l.append([i,xc,yc])
    
    return l



def run_mtcnn(det, list_frames):

    '''
    det : detector
    list_frames : list of frames to infer
    
    '''

    All_kpts = []
    All_boxes = []

    for f in list_frames:

        frame_boxes, frame_kpts = run_detection(det, f)
        frame_normalized_kpts, frame_boxes = normalize_bbox_kpts(frame_boxes, frame_kpts)

        All_kpts.append(frame_normalized_kpts)
        All_boxes.append(frame_boxes)
        
    return All_kpts, All_boxes



def get_face_score(All_kpts, label_kpts, All_boxes):

    '''
    find the highest score frame in the list and get its score
    get also bounding boxes of the frame
    '''

    list_scores = []

    for frame_norm_kpts , frame_boxes in zip(All_kpts, All_boxes):

        scores = score_face(label_kpts , frame_norm_kpts, frame_boxes)
        median = np.mean(scores)
        list_scores.append(median)

    # max_score = np.max(LIST_SCORES)
    frame_kpts = All_kpts[np.argmax(list_scores)]
    frame_boxes = All_boxes[np.argmax(list_scores)]
    index = np.argmax(list_scores)

    scores  = score_face(label_kpts , frame_kpts, frame_boxes)

    return scores, frame_boxes, index


def face_bad_scores_box(frame_boxes, scores, frame):

    '''
    return bad smile in the frame
    '''

    box = frame_boxes[np.argmin(scores)]

    # return [box]

    x,y,x2,y2 = box
    x = int(x)
    y = int(y)
    x2 = int(x2)
    y2 = int(y2)

    bim = frame[y:y2,x:x2]

    return bim

