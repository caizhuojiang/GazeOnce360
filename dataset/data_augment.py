import cv2
import numpy as np
import random

def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)

class preproc(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes_t = targets[:, :4].copy()
        landm_t = targets[:, 4:28].copy()
        gaze_t  = targets[:, 28:31].copy()
        hdps_t = targets[:, 31:34].copy()
        dist_t = targets[:, 34:35].copy()
        labels_t = targets[:, 35].copy()
        assert(targets.shape[1] == 36)
        
        image_t = image.copy()

        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, gaze_t, hdps_t, dist_t, labels_t))

        return image_t, targets_t
