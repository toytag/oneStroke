import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order

img = cv2.imread("portraits/theLady.jpeg", 0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.1, 4)
(x, y, w, h) = faces[np.argmax(faces[:,-1])]

crop_img = cv2.resize(img[int(y-0.1*h):int(y+1.1*h), int(x-0.1*w):int(x+1.1*w)], (200, 200))

blur_img = cv2.bilateralFilter(crop_img, 7, 50, 50)

edge_img = cv2.Canny(blur_img, 30, 60)

edge_indexs = np.argwhere(edge_img != 0)
edge_points = np.zeros(edge_indexs.shape)
edge_points[:,0] = edge_indexs[:,1]
edge_points[:,1] = 200 - edge_indexs[:,0]

n = len(edge_points)

adjMat = np.zeros((n, n))
for i in range(n):
    adjMat[i, :] = np.linalg.norm(edge_points - edge_points[i], axis=1)
adjMat = csr_matrix(adjMat)
adjMatMst = minimum_spanning_tree(adjMat)

i_start = divmod(np.argmax(adjMatMst), len(edge_points))[0]
path, predecessors = depth_first_order(adjMatMst, i_start, directed=False)

fp = []
for i in range(len(path)-1):
    curVertex = path[i]
    fp.append(curVertex)
    nextVertex = path[i+1]
    while predecessors[nextVertex] != curVertex:
        curVertex = predecessors[curVertex]
        fp.append(curVertex)
fp.append(path[-1])

def RamerDouglasPeucker(path, epsilon=2):
    start = 0
    end = len(path) - 1
    vec1 = (edge_points[path[end]] - edge_points[path[start]]).reshape(-1,1)
    vecs = edge_points[path] - edge_points[path[start]]
    projMat = vec1.dot(vec1.T) / np.linalg.norm(vec1) ** 2
    errVec = (np.eye(2) - projMat).dot(vecs.T)
    errNorm = np.linalg.norm(errVec.T, axis=1)
    imax = np.argmax(errNorm)
    dmax = errNorm[imax]
    if dmax >= epsilon:
        return RamerDouglasPeucker(path[start:imax+1]) + RamerDouglasPeucker(path[imax:end+1])[1:]
    else:
        return [path[start], path[end]]

downFp = RamerDouglasPeucker(fp)

print(downFp)