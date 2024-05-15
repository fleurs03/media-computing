import numpy as np

def get_bounding_rect(object):
    # black is 0, white is 255
    # black background, white object
    for i in range(object.shape[0]):
        if np.sum(object[i]) > 0:
            top_bound = i
            break

    for i in range(object.shape[0]-1, -1, -1):
        if np.sum(object[i]) > 0:
            bottom_bound = i
            break

    for i in range(object.shape[1]):
        if np.sum(object[:,i]) > 0:
            left_bound = i
            break

    for i in range(object.shape[1]-1, -1, -1):
        if np.sum(object[:,i]) > 0:
            right_bound = i
            break
    return top_bound, bottom_bound, left_bound, right_bound
