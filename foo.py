# This Python file uses the following encoding: utf-8
import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from oneStroke import MODEL_DIR


def face_crop(img):
    """
    Find the face segement of an image.

    Parameters
    ----------
    `img` : 2d numpy array 
        grayscale image

    Returns
    -------
    `face_crop` : 2d numpy array 
        grayscale image resized to 256x256

    """
    face_cascade = cv2.CascadeClassifier(MODEL_DIR)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    (x, y, w, h) = faces[np.argmax(faces[:,-1])]
    face_crop = cv2.resize(img[int(y-0.1*h):int(y+1.1*h), int(x-0.1*w):int(x+1.1*w)], (256, 256))
    return face_crop


def edge_detect(img):
    """
    Find the edge points of an image.

    Parameters
    ----------
    `img` : 2d numpy array 
        grayscale image

    Returns
    -------
    `edge_points` : 2d numpy array 
        a set of (x, y) coordinates

    """
    blur_img = cv2.bilateralFilter(img, 7, 50, 50)
    edge_img = cv2.Canny(blur_img, 30, 60)
    edge_indexs = np.argwhere(edge_img != 0)
    edge_points = np.zeros(edge_indexs.shape)
    edge_points[:,0] = edge_indexs[:,1]
    edge_points[:,1] = img.shape[0] - edge_indexs[:,0]
    return edge_points


def adj_matrix(points):
    """
    Find the adjacency matrix of a complete graph defined by a set of points.

    Parameters
    ----------
    `points` : 2d numpy array 
        a set of (x, y) coordinates

    Returns
    -------
    `adjMat` : scipy sparse matrix 
        adjacency matrix of a complete graph defined by `points`

    """
    n = points.shape[0]
    adjMat = np.zeros((n, n))
    for i in range(n):
        adjMat[i, :] = np.linalg.norm(points - points[i], axis=1)
    adjMat = csr_matrix(adjMat)
    return adjMat


def mst(adjMat):
    """
    Find the minimum spanning tree of a graph defined by adjacency matrix.

    Parameters
    ----------
    `adjMat` : scipy sparse matrix
        adjacency matrix of a graph

    Returns
    -------
    `adjMat_of_mst` : scipy sparse matrix 
        adjacency matrix of minimum spanning tree of the graph

    """
    adjMat_of_mst = minimum_spanning_tree(adjMat)
    return adjMat_of_mst


def dfs(adjMat):
    """
    Find the depth first search order of a graph defined by adjacency matrix.

    Parameters
    ----------
    `adjMat` : scipy sparse matrix
        adjacency matrix of a graph

    Returns
    -------
    `target_path` : 1d numpy array 
        a path of indices of points that walk through the graph in depth first order

    """
    # need improvement to 减少回头路
    i_start = divmod(np.argmax(adjMat), adjMat.shape[0])[0]
    path, predecessors = depth_first_order(adjMat, i_start, directed=False)
    target_path = []
    for i in range(len(path) - 1):
        curVertex = path[i]
        target_path.append(curVertex)
        nextVertex = path[i+1]
        while predecessors[nextVertex] != curVertex:
            curVertex = predecessors[curVertex]
            target_path.append(curVertex)
    target_path.append(path[-1])
    target_path = np.array(target_path)
    return target_path


def rdp(points, path, epsilon=2):
    """
    Use [Ramer–Douglas–Peucker algorithm](https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm)
    to downsample points on path.

    Parameters
    ----------
    `points` : 2d numpy array 
        a set of (x, y) coordinates
    `path` : 1d numpy array 
        a path of indices of points
    `epsilon` : positive real number
        maximum tolerance of error

    Returns
    -------
    Downsampled path

    Note
    ------
    Sometimes, if the path goes back and forth, the implementaion may not work properly.
    Say if the path starts at one point, goes wherever it want and ends at the exact
    point it starts, the implementation would just give you that point.

    """
    start = 0
    end = len(path) - 1
    vec1 = (points[path[end]] - points[path[start]]).reshape(-1,1)
    vecs = points[path] - points[path[start]]
    projMat = vec1.dot(vec1.T) / np.linalg.norm(vec1) ** 2
    errVec = (np.eye(2) - projMat).dot(vecs.T)
    errNorm = np.linalg.norm(errVec.T, axis=1)
    imax = np.argmax(errNorm)
    dmax = errNorm[imax]
    if dmax >= epsilon:
        return np.hstack((rdp(points, path[start:imax+1]), rdp(points, path[imax:end+1])[1:]))
    else:
        return np.array([path[start], path[end]])
