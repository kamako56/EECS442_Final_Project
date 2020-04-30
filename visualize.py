import os
import time
import trimesh
import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
from KPE import KPE
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from skimage import measure
from dataset import PeopleDataset
from Model import Version1,Version2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=4
train_data = PeopleDataset(flag='', data_range=(2000,2500))

net = Version1().to(device)
net.load_state_dict(torch.load('models/model_experiment1.pth'))
imgs=[]
imgs.append(train_data[7][1])
imgs.append(train_data[165][1])
imgs.append(train_data[165+125][1])
imgs.append(train_data[165+125*2][1])

from mayavi import mlab
for img in imgs:
    # voxels=net(img.unsqueeze(0).cuda())[3][0].cpu().detach().numpy()
    voxels=img.detach().cpu().numpy()
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxels,level=0.1,step_size=1,spacing=(16.,16.,16.)) # doctest: +SKIP
    mlab.triangular_mesh([vert[0] for vert in verts],
                            [vert[1] for vert in verts],
                            [vert[2] for vert in verts],
                            faces) # doctest: +SKIP
    mlab.show()