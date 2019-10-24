# This Python file uses the following encoding: utf-8

# may not function on mac if you use anaconda
# 这个在Mac上用anaconda居然用不了
# 很可能会收到如下报错
# you might get error like this

# from matplotlib.backends import _macosx
# RuntimeError: Python is not installed as a framework. The Mac OS X backend will 
# not be able to function correctly if Python is not installed as a framework. See
# the Python documentation for more information on installing Python as a framework
# on Mac OS X. Please either reinstall Python as a framework, or try one of the 
# other backends. If you are using (Ana)Conda please install python.app and replace 
# the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the 
# Matplotlib FAQ for more information.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from oneStroke import *

img = cv2.imread(sys.argv[1], 0)
edge_points = edge_detect(face_seg(img))
path = dfs(mst(adj_matrix(edge_points)))
rdp_path = rdp(edge_points, path)

x = edge_points[:,0].flatten()
y = edge_points[:,1].flatten()
edges = np.hstack((rdp_path[:-1].reshape(-1,1), rdp_path[1:].reshape(-1,1)))

plt.axis('equal')
plt.plot(x[edges.T], y[edges.T], c='tab:blue')
plt.show()

# 用不了我也没有试过。。。。。。等我能试再看看行不行
# 从notebook里复制过来的应该没啥问题吧