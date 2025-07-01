import torch
from torch.nn import functional as F
import cv2
import random
import numpy as np

'''
enum InterpolationFlags
{
    ``'bilinear'`` | ``'bicubic'`` | ``'area'``
};
'''


def random_resizing(image,updown_type,resize_prob,mode_list,resize_range):
    b, c, h, w= image.shape

    updown_type = random.choices(updown_type, resize_prob)[0]   #choices返回list ["up"],所以要通过 [0] 取list第一个元素
    mode = random.choice(mode_list)

    if updown_type == "up":
        scale = np.random.uniform(1, resize_range[1])
    elif updown_type == "down":
        scale = np.random.uniform(resize_range[0], 1)
    else:
        scale = 1

    image = F.interpolate(image,scale_factor=scale,mode=random.choice(['area','bilinear','bicubic']))
    #image = cv2.resize(image, (w, h), interpolation=flags)
    image = torch.clamp(image, 0.0, 1.0)
    return image
