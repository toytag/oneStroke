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

from .foo import face_seg, edge_detect, mst, dfs, mst
__all__ = ["face_seg", "edge_detect", "mst", "dfs", "mst"]