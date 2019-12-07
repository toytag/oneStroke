# This Python file uses the following encoding: utf-8
from __future__ import division, absolute_import, print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
from oneStroke import *

import sys
img_path = sys.argv[1]
del sys

img = cv2.imread(img_path, 0)

# gamma correction
# gamma = 1.7
# img = img ** (1 / gamma)
# cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX)
# img = np.uint8(img)

# histgram equalization
# img = cv2.equalizeHist(img)

# CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img = clahe.apply(img)

# print(img)

cv2.imwrite("new.png", img)
print("--- picture \"new.png\" saved ---")
# exit(0)

edge_points = edge_detect(face_crop(img))
x, y = edge_points.T
plt.cla()
plt.axis("equal")
plt.scatter(x, y, s=2)
plt.savefig("edge.png")
print("--- picture \"edge.png\" saved ---")

path_index = dfs(mst(adj_matrix(edge_points)))
path = edge_points[path_index]
x, y = path.T
plt.cla()
plt.axis("equal")
plt.plot(x, y)
plt.savefig("path.png")
print("--- picture \"path.png\" saved ---")

downsampled_path = rdp(path)
print(len(downsampled_path))
x, y = downsampled_path.T
plt.cla()
plt.axis("equal")
plt.plot(x, y, linewidth=2.56)
plt.savefig("rdp.png")
print("--- picture \"rdp.png\" saved ---")
