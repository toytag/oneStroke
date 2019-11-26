import sys, cv2
from oneStroke import *

img_path = sys.argv[1]

img = cv2.imread(img_path, 0)
edge_points = edge_detect(face_crop(img))
path_index = dfs(mst(adj_matrix(edge_points)))
path = edge_points[path_index]

print("len of original path: {}".format(len(path)))
for eps in [0.5, 1, 1.5, 2, 2.56, 3, 5, 10]:
    downsample_path = rdp(path, epsilon=eps)
    print("eplison = {} \t len of downsampled path: {}".format(eps, len(downsample_path)))