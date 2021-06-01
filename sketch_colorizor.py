from operator import ne
import cv2
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import time

def colorize(img, sketch):
    
    def rgb2yiq(rgb):
        rgb = rgb / 255.0
        y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])),             0,   1)
        i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
        q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])),  -0.5226, 0.5226)
        yiq = rgb
        yiq[..., 0] = y
        yiq[..., 1] = i
        yiq[..., 2] = q
        return yiq


    def yiq2rgb(yiq):
        r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
        g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
        b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
        rgb = yiq
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        out = np.clip(rgb, 0.0, 1.0) * 255.0
        out = out.astype(np.uint8)
        return out

    def cal_index(img, p):
        row_length = img.shape[1]
        return p[0] * row_length + p[1]

    # transfer from BGR to YUV
    img = rgb2yiq(img)
    sketch = rgb2yiq(sketch)

    n, m = img.shape[0], img.shape[1]
    colored_region = sketch.sum(2)
    colored_indices = cal_index(img, np.nonzero(colored_region))
    # TODO: 处理白色区域

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
        img_lum = np.array(img[:, :, 0], dtype=np.float64)
        neighbor_idx = [cal_index(img, p) for p in neighbors]
        neighbor_lum = [img_lum[p[0], p[1]] for p in neighbors]
        neighbor_std = np.std(neighbor_lum)
        # some special tricks for std
        if (neighbor_std <= 0):
            return np.zeros(shape=len(neighbors)), neighbor_idx
        else:
            return [np.exp(-1 * np.square(img_lum[tuple(p)] - y) / 2 / neighbor_std / neighbor_std) for y in neighbor_lum], neighbor_idx
    
    def get_weight_matrix(img):
        n, m = img.shape[0], img.shape[1]
        weight_matrix = sparse.lil_matrix((n * m, n * m))

        for i in range(n):
            for j in range(m):
                neighbor_weights, neighbor_idx = get_weights(img, [i, j])
                weight_matrix[cal_index(img, [i, j]), neighbor_idx] = -1 * np.asarray(neighbor_weights)

        weight_matrix = normalize(weight_matrix, norm='l1', axis=1).tolil()
        weight_matrix[np.arange(n * m), np.arange(n * m)] = 1.
        return weight_matrix
    
    weight_matrix = get_weight_matrix(img)
    print("finish calculating matrix")
    
    weight_matrix = weight_matrix.tocsc()
    start = time.time()

    for p in list(colored_indices):
        weight_matrix[p] = sparse.csr_matrix(([1.0], ([0], [p])), shape=(1, n*m))
    
    b1 = np.zeros(m * n)
    b2 = np.zeros(m * n)
    b1[colored_indices] = sketch[:, :, 1].flatten()[colored_indices]
    b2[colored_indices] = sketch[:, :, 2].flatten()[colored_indices]

    x1 = sparse.linalg.spsolve(weight_matrix, b1)
    x2 = sparse.linalg.spsolve(weight_matrix, b2)
    print("finish calculating")
    print(time.time() - start)

    output = np.zeros((n, m, 3), dtype=np.float64)
    output[:, :, 0] = img[:, :, 0]
    output[:, :, 1] = x1.reshape((n, m)) 
    output[:, :, 2] = x2.reshape((n, m))
    output = yiq2rgb(output)   

    cv2.imshow("title", output)
    cv2.waitKey(0)
        