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
from Model import Version2

def cal_IOU(testloader, net, device,threshold):
    '''
    Calculate IOU
    '''
    ious=[]
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[3]
            binary_output = torch.zeros_like(output)
            binary_output[output>threshold]=1
            binary_output[output<threshold]=0
            intersection=binary_output==labels
            intersection[binary_output==0]=0
            union = binary_output
            union[labels==1]=1
            for i in range(images.shape[0]):
                ious.append((torch.sum(intersection[i])/(torch.sum(union[i]))).detach().cpu().numpy())

    return np.mean(ious)
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size=2
    train_data = PeopleDataset(flag='', data_range=(0,2000))
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    val_data = PeopleDataset(flag='', data_range=(2000,2375))
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True)
    test_data = PeopleDataset(flag='', data_range=(2375,2500))
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)
    net = Version2().to(device)
    net.load_state_dict(torch.load('models/model_experiment5_epoch43.pth'))

    print("----------Train Data------------")
    for level in np.linspace(0.1,1,10):
        threshold = level
        print("Threshold:",threshold,"IOU:",cal_IOU(train_loader,net,device,threshold))
    print("----------Val Data------------")
    for level in np.linspace(0.1,1,10):
        threshold = level
        print("Threshold:",threshold,"IOU:",cal_IOU(val_loader,net,device,threshold))
    print("----------Test Data------------")
    for level in np.linspace(0.1,1,10):
        threshold = level
        print("Threshold:",threshold,"IOU:",cal_IOU(test_loader,net,device,threshold))
    # val_data = FacadeDataset(flag='train', data_range=(805,906), onehot=False)
    # val_loader = DataLoader(val_data, batch_size=batch_size)
    # test_data = FacadeDataset(flag='', data_range=(0,114))
    # test_loader = DataLoader(test_data, batch_size=1)

    # ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    # ap_loader = DataLoader(ap_data, batch_size=1)


    # for threshold in np.linspace(0.1,1,num=10):
    #     print(cal_IOU(train_loader,net,device,threshold))
    img,voxels_true=train_data[0]
    voxels=net(img.unsqueeze(0).cuda())[3][0].cpu().detach().numpy()
    from mayavi import mlab
    # binary_voxels=np.zeros_like(voxels)
    # binary_voxels[voxels>threshold]=1
    # binary_voxels[voxels<threshold]=0
    # xx, yy, zz = np.where(binary_voxels == 1)
    # mlab.points3d(xx, yy, zz,
    #                      mode="cube",
    #                      color=(0, 1, 0),
    #                      scale_factor=1)
    # mlab.show()
    # xx, yy, zz = np.where(voxels_true == 1)
    # mlab.points3d(xx, yy, zz,
    #                      mode="cube",
    #                      color=(0, 1, 0),
    #                      scale_factor=1)
    # mlab.show()
    for level in np.linspace(0.1,1,10):
        threshold = level
        print(cal_IOU(train_loader,net,device,threshold))
        verts, faces, normals, values = measure.marching_cubes_lewiner(voxels,level=level,step_size=1,spacing=(16.,16.,16.)) # doctest: +SKIP
        mlab.triangular_mesh([vert[0] for vert in verts],
                                [vert[1] for vert in verts],
                                [vert[2] for vert in verts],
                                faces) # doctest: +SKIP
        # verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_true.detach().cpu().numpy(),level=level,step_size=1,spacing=(16.,16.,16.)) # doctest: +SKIP
        # mlab.triangular_mesh([vert[0] for vert in verts],
        #                         [vert[1] for vert in verts],
        #                         [vert[2] for vert in verts],
        #                         faces) # doctest: +SKIP
        mlab.show() # doctest: +SKIPa