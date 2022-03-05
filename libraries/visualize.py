
import numpy as np
import cv2


l_pair = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16)
]
p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
            (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
            (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]


def process_result(res):


    X = []
    Y = []
    SCORE = []

    for p in res:
        x = []
        y = []
        score = []

        for i, el in enumerate(p['keypoints']):
            if i%3 == 0:
                x.append(el)
            if i%3 == 1:
                y.append(el)
            if i%3 == 2:
                score.append(el)

        # newx = (x[5]+x[6])/2
        # newy = y[5]
        # x.insert(6,newx)
        # y.insert(6,newy)
        # score.insert(6,0.6)

        X.append(x)
        Y.append(y)
        SCORE.append(score)

    return X, Y, SCORE

def vis_keypoints_jointlines(frame, res, X, Y, SCORE, on_frame = 0):

    if on_frame == 0:
        img = np.full((frame.shape[0], frame.shape[1],3),255).astype(np.uint8)
    elif on_frame == 1:
        img = frame.copy()
    
    for i, _ in enumerate(res):

        x = X[i]
        y = Y[i]
        score = SCORE[i]
        part_line = {}
        # Draw keypoints
        for n in range(len(x)):
            # if score[n] <= 0.35:
            #     continue
            cor_x, cor_y = int(x[n]), int(y[n])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255,255,255), 2)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(score[start_p] + score[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255,255,255), 1)  

    return img