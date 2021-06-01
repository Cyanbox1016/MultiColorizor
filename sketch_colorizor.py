from operator import ne
import cv2
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import time

def colorize(img, sketch):
    # transfer from BGR to YUV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2YUV)

    img = img.astype(np.float64)
    sketch = sketch.astype(np.float64)

    # prepare for output
    output = np.zeros(img.shape, dtype=np.float64)
    output[:, :, 0] = img[:, :, 0]
    output[:, :, 1] = sketch[:, :, 1]
    output[:, :, 2] = sketch[:, :, 2]

    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_YUV2BGR)

    cv2.imshow("title", output)
    cv2.waitKey(0)

    start = time.time()

    def cal_index(img, p):
        row_length = img.shape[1]
        return p[0] * row_length + p[1]

    def get_neighbors(img, p, d=3):
        y = p[0]
        x = p[1]
        left = max(0, x - d)
        right = min(img.shape[1] - 1, x + d)
        top = max(0, y - d)
        bottom = min(img.shape[0] - 1, y + d)
        neighbors = []
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                if not (i == y and j == x):
                    neighbors.append((i, j))

        return neighbors
    
    def get_weights(img, p):
        neighbors = get_neighbors(img, p)
        img_lum = img[:, :, 0]
        neighbor_idx = [cal_index(img, p) for p in neighbors]
        neighbor_lum = [img_lum[p[0], p[1]] for p in neighbors]
        neighbor_std = np.std(neighbor_lum)
        
        # some special tricks for std
        sigma = neighbor_std * 0.75
        mgv = min((neighbor_lum - img_lum[p[0], p[1]]) ** 2)
        sigma = max(sigma, -mgv / np.log(0.01))
        sigma = max(sigma, 0.000002)
        neighbor_weights = np.exp(-(neighbor_lum - img_lum[p[0], p[1]]) ** 2 / (2 * (sigma) ** 2))
        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
        return neighbor_weights, neighbor_idx
    
    def get_weight_matrix(img):
        n, m = img.shape[0], img.shape[1]
        weight_matrix = sparse.lil_matrix((n * m, n * m))
        print(weight_matrix.shape)

        for i in range(n):
            for j in range(m):
                neighbor_weights, neighbor_idx = get_weights(img, (i, j))
                weight_matrix[cal_index(img, (i, j)), neighbor_idx] = neighbor_weights
        weight_matrix[np.arange(n * m), np.arange(n * m)] = 1.
        return weight_matrix
    
    weight_matrix = get_weight_matrix(img)
    print("finish calculating matrix")
    print(time.time() - start)

    
    weight_matrix = weight_matrix.tocsc()
    colored_region = sketch.sum(2) > 2
    colored_indices = np.nonzero(colored_region.reshape(img.shape[0] * img.shape[1], order='F'))

    for channel in [1, 2]:
        colored_img = sketch[:, :, channel].flatten()
        b = np.zeros(img.shape[0] * img.shape[1])
        b[colored_indices] = colored_img[colored_indices]
        x = sparse.linalg.spsolve(weight_matrix, b)[:img.shape[0]*img.shape[1]]
        output[:, :, channel] = x.reshape(img.shape[0], img.shape[1])
    
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_YUV2BGR)

    cv2.imshow("title", output)
    cv2.waitKey(0)
        