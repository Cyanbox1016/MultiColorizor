import numpy as np
import cv2
from random import randint

class RecRegion(object):

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x_start = x
        self.y_start = y
        self.w = w
        self.h = h

        self.x_end = x + w
        self.y_end = y + h


def colorize_inside_region(img, ref, img_region: RecRegion, ref_region: RecRegion):
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
    ref_local = ref[ref_region.y_start:ref_region.y_end, ref_region.x_start:ref_region.x_end, :]
    img_local = img[img_region.y_start:img_region.y_end, img_region.x_start:img_region.x_end]

    cv2.imshow("title", cv2.cvtColor(ref_local, cv2.COLOR_LAB2BGR))
    cv2.waitKey(0)

    miu_ref = np.mean(ref_local[:, :, 0])
    miu_img = np.mean(img_local)

    sigma_ref = np.std(ref_local[:, :, 0])
    sigma_img = np.std(img_local)

    ref_local[:, :, 0] = (ref_local[:, :, 0] - miu_ref) * sigma_img / sigma_ref + miu_img

    sampled = []
    window_size = 5
    y_num = ref_local.shape[0] // window_size + 1
    x_num = ref_local.shape[1] // window_size + 1

    for i in range(y_num):
        for j in range(x_num):
            pos_y = randint(0, window_size - 1)
            pos_x = randint(0, window_size - 1)
            pos_y = min(pos_y + i * window_size, ref_local.shape[0] - 1)
            pos_x = min(pos_x + j * window_size, ref_local.shape[1] - 1)
            sampled.append(ref_local[pos_y, pos_x, :])

    img_generated = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2Lab)
    for i in range(img_local.shape[0]):
        for j in range(img_local.shape[1]):
            ori_i = i + img_region.y_start
            ori_j = j + img_region.x_start
            lum_local = int(img_local[i, j])
            lum_sample = int(sampled[0][0])
            img_generated[ori_i, ori_j, 1] = sampled[0][1]
            img_generated[ori_i, ori_j, 2] = sampled[0][2]
            l_diff = abs(lum_local - lum_sample)
            img_generated[ori_i, ori_j, 0] = img_local[i, j]
            for pixel in sampled:
                lum_sample = int(pixel[0])
                if (abs(lum_local - lum_sample) <= l_diff):
                    l_diff = abs(lum_local - lum_sample)
                    img_generated[ori_i, ori_j, 1] = pixel[1]
                    img_generated[ori_i, ori_j, 2] = pixel[2]

    img_generated = cv2.cvtColor(img_generated, cv2.COLOR_LAB2BGR)
    cv2.imshow("title", img_generated)
    cv2.waitKey(0)

