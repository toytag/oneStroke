# oneStroke

Use *oneStroke drawing*, to draw a portrait. 

## Description

All the implementations are in `foo.py`. 

Unfortunately this implementation does not find a Euler path, even if the graph is somehow Eulerian graph. (by somehow I mean almost impossible, unless magic) 

### GLHF Floder

- `GLHF/oneStroke.ipynb`

    `jupyter` and `matplotlib` is also required if you want to dive in `GLHF/oneStroke.ipynb`. This jupyter notebook can help you getting a better understanding of this specific implementation.

- `GLHF/FCNberkeley.ipynb`

    `jupyter`, `matplotlib` and `caffe` is also required if you want to explore this more. FCN stands for **Fully Convolutional Network**, originally proposed in the [PAMI FCN](https://arxiv.org/abs/1605.06211) and [CVPR FCN](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) papers, to solve image segmentation problems. This specific model used in this notebook is FCN-8s which can be found in [BerkeleyVision](https://github.com/shelhamer/fcn.berkeleyvision.org) and it is trained on [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

## Dependencies

```pip install -r requirements.txt```

Basically, it's just `opencv-python`, `numpy`, `scipy`. Not too many dependencies. 

## Last But Not The Least
Good Luck Have Fun :)
