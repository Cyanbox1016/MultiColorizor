import numpy as np
import cv2
from scipy.ndimage import generic_filter
from random import randint


class RectRegion(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x_start = x
        self.y_start = y
        self.w = w
        self.h = h

        self.x_end = x + w
        self.y_end = y + h


def colorize_swatch(img, ref):
    cv2.imshow("title", ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
    print(ref.shape)

    ref = ref.astype(np.float64)
    ref_lum = ref[:, :, 0]

    # luminance remapping
    ref_lum_mean = np.mean(ref_lum)
    img_mean = np.mean(img)
    ref_lum_std = np.std(ref_lum)
    img_std = np.std(img)

    ref_lum = (ref_lum - ref_lum_mean) * img_std / ref_lum_std + img_mean

    # calculate neighborhood statistics
    neighborhood_size = 5
    ref_stat = generic_filter(ref_lum, np.std, neighborhood_size)
    img_stat = generic_filter(img, np.std, neighborhood_size)

    # jittered sample
    window_size = 5
    grid_num_x = int(ref.shape[1] / window_size)
    grid_num_y = int(ref.shape[0] / window_size)

    sampled_pos = []
    sampled_pixel = []
    sampled_neighbor_stat = []

    for i in range(grid_num_y):
        for j in range(grid_num_x):
            pos_y = randint(i * window_size, (i + 1) * window_size - 1)
            pos_x = randint(j * window_size, (j + 1) * window_size - 1)
            sampled_pos.append((pos_y, pos_x))
            sampled_pixel.append(ref_lum[pos_y][pos_x])
            sampled_neighbor_stat.append(ref_stat[pos_y][pos_x])

    # find best match
    output = np.full((img.shape[0], img.shape[1], 3), 128, dtype=np.float64)
    output[:, :, 0] = img

    weight_lum = 0.5
    weight_stat = 1. - weight_lum
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            weighted_sum = (weight_lum * np.square(sampled_pixel - img[i][j])) + \
                           (weight_stat * np.square(sampled_neighbor_stat - img_stat[i][j]))
            match_index = np.argmin(weighted_sum)
            [ref_y, ref_x] = sampled_pos[match_index]
            output[i][j][1] = ref[ref_y][ref_x][1]
            output[i][j][2] = ref[ref_y][ref_x][2]

    output = output.astype(np.uint8)
    # output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)

    return output


def colorize_photo(img, ref, img_swatches: list, ref_swatches: list):
    assert len(img_swatches) == len(ref_swatches)
    mask = np.zeros(img.shape, dtype=np.int)
    swatch_num = len(img_swatches)
    output = np.full((img.shape[0], img.shape[1], 3), 128, dtype=np.uint8)
    output[:, :, 0] = img

    d = 2
    colorized_swatches = []

    for idx in range(0, swatch_num):
        img_region = img_swatches[idx]
        ref_region = ref_swatches[idx]
        # assert img_region is RectRegion and ref_region is RectRegion
        img_swatch = img[img_region.y_start:img_region.y_end + 1, img_region.x_start:img_region.x_end + 1]
        ref_swatch = ref[ref_region.y_start:ref_region.y_end + 1, ref_region.x_start:ref_region.x_end + 1]
        output_swatch = output[img_region.y_start:img_region.y_end + 1, img_region.x_start:img_region.x_end + 1]
        output_swatch[:, :, :] = colorize_swatch(img_swatch, ref_swatch)
        mask[img_region.y_start:img_region.y_end + 1, img_region.x_start:img_region.x_end + 1] = 1
        colorized_swatches.append(cv2.copyMakeBorder(output_swatch, d * 2, d * 2,
                                                     d * 2, d * 2, cv2.BORDER_REFLECT))

    # expand color to the rest part
    output = cv2.copyMakeBorder(output, d * 2, d * 2,
                                d * 2, d * 2, cv2.BORDER_REFLECT)
    mask = cv2.copyMakeBorder(output, d * 2, d * 2,
                              d * 2, d * 2, cv2.BORDER_REFLECT)

    for i in range(2 * d, output.shape[0], 2 * d):
        if i >= output.shape[0] - d:
            continue
        for j in range(2 * d, output.shape[1], 2 * d):
            if j >= output.shape[1] - d:
                continue
            if np.sum(mask[i-d:i+d, j-d:j+d]) == np.square(2 * d + 1):
                continue
            patch = np.copy(output[i-d:i+d, j-d:j+d])
            patch = patch.astype(np.float64)
            error = 1e8

            for swatch in colorized_swatches:
                for m in range(2 * d, swatch.shape[0], 2 * d):
                    if m >= swatch.shape[0] - d:
                        continue
                    for n in range(2 * d, swatch.shape[1], 2 * d):
                        if n >= swatch.shape[1] - d:
                            continue
                        swatch_patch = np.copy(swatch[m-d:m+d, n-d:n+d])
                        swatch_patch = swatch_patch.astype(np.float64)
                        distance = np.sum(np.square(swatch_patch - patch))
                        if (distance < error):
                            error = distance
                            output[i-d:i+d, j-d:j+d, 1:] = swatch[m-d:m+d, n-d:n+d, 1:]

    output = output[2*d:output.shape[0]-2*d, 2*d:output.shape[1]-2*d]
    output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)

    cv2.imshow("title", output)
    cv2.waitKey(0)
