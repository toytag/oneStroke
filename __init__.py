# This Python file uses the following encoding: utf-8
from __future__ import division, absolute_import, print_function

import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'haarcascade_frontalface_default.xml')
if not os.path.exists(MODEL_DIR):
    msg = """ 
Face detection model <haarcascade_frontalface_default.xml> does not exist.
Fail to import oneStroke.
    """
    raise ImportError(msg)

del os

from .foo import face_crop, edge_detect, adj_matrix, mst, dfs, rdp
__all__ = ['face_crop', 'edge_detect', 'adj_matrix', 'mst', 'dfs', 'rdp']
