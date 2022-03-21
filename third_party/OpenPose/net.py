

import torch
from third_party.OpenPose.modules.load_state import load_state
from third_party.OpenPose.models.with_mobilenet import PoseEstimationWithMobileNet


def load_model():

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('third_party/OpenPose/checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)

    print('model loaded')

    return net
