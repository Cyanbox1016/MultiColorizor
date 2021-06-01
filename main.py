import numpy as np
import cv2
from local_tinter import RecRegion, colorize_inside_region
from swatch_colorizor import colorize_photo
from sketch_colorizor_2 import colorize

# img = cv2.imread("4.jpg")
# ref = cv2.imread("3.png")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_swatch = RecRegion(0, 0, 200, 50)
# ref_swatch = RecRegion(0, 0, 350, 100)

# img_swatch2 = RecRegion(0, 200, 200, 50)
# ref_swatch2 = RecRegion(0, 300, 350, 100)

# img_l = [img_swatch, img_swatch2]
# ref_l = [ref_swatch, ref_swatch2]

# colorize_photo(img, ref, img_l, ref_l)

img = cv2.imread("x1.jpg")
sketch = cv2.imread("k1.bmp")
cv2.imshow("title", img)
cv2.waitKey(0)

# cv2.imshow("title", sketch)
# cv2.waitKey(0)
colorize(img, sketch)
