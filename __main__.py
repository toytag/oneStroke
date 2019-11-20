import sys, cv2
from oneStroke import *

img_path = sys.argv[1]

img = cv2.imread(img_path, 0)
edge_points = edge_detect(face_crop(img))
path = dfs(mst(adj_matrix(edge_points)))
downsample_path = rdp(edge_points, path, epsilon=2)

print(edge_points[downsample_path])