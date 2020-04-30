import os
import torch
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "egl"
from KPE import KPE
import numpy as np
import numpy.linalg
import pyrender
import trimesh
import matplotlib.pyplot as plt
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            dump=scene_or_mesh.dump()
            mesh=dump[0]
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh
# generate mesh
def pad_voxelization(mat,xsize,ysize,zsize):
    voxels=mat.copy()
    xpad=max(xsize-voxels.shape[0],0)
    ypad=max(ysize-voxels.shape[1],0)
    zpad=max(zsize-voxels.shape[2],0)
    voxels = np.concatenate((np.zeros((xpad//2,voxels.shape[1],voxels.shape[2])),voxels,np.zeros((xpad-(xpad//2),voxels.shape[1],voxels.shape[2]))),axis=0)
    voxels = np.concatenate((np.zeros((voxels.shape[0],ypad//2,voxels.shape[2])),voxels,np.zeros((voxels.shape[0],ypad-(ypad//2),voxels.shape[2]))),axis=1)
    voxels = np.concatenate((np.zeros((voxels.shape[0],voxels.shape[1],zpad//2)),voxels,np.zeros((voxels.shape[0],voxels.shape[1],zpad-zpad//2))),axis=2)
    return voxels
# tmesh=as_mesh(trimesh.load('untitled.glb'))
# tmesh.rezero()
# centroid=tmesh.centroid
# tmesh.apply_translation(-1*centroid)
# tmesh.apply_scale(1/tmesh.scale)
# voxels=tmesh.voxelized(1/224)
# print(voxels.matrix.shape)
# mat=pad_voxelization(voxels.matrix,128,256,96)
# print(mat.shape)
# #tmesh.show()
# #voxels.show()
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.voxels(voxels.matrix , edgecolor='k')
# # plt.show()
# mesh = pyrender.Mesh.from_trimesh(tmesh, smooth=True)


# compose scene
mintheta=-np.pi/4
maxtheta=np.pi/4
minphi=-np.pi/8
maxphi=np.pi/16
anglesteps=5
mindist=1
maxdist=1.5
diststeps=5
i=0
for mm in range(2,26):
    if mm == 5 or mm == 14 or mm == 17 or mm == 20 or mm == 14:
        continue
    tmesh=as_mesh(trimesh.load('mm'+str(mm)+'.glb'))
    tmesh.rezero()
    centroid=tmesh.centroid
    tmesh.apply_translation(-1*centroid)
    tmesh.apply_scale(1/tmesh.scale)
    voxels=tmesh.voxelized(1/196)
    # voxels=voxels.fill()
    # mat=pad_voxelization(voxels.matrix,128,256,96)
    #tmesh.show()
    # voxels.show()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(voxels.matrix , edgecolor='k')
    # plt.show()
    mesh = pyrender.Mesh.from_trimesh(tmesh, smooth=True)
    for meandist in np.linspace(mindist,maxdist,diststeps):
        for meantheta in np.linspace(mintheta,maxtheta,anglesteps):
            for meanphi in np.linspace(minphi,maxphi,anglesteps):
                theta=meantheta+np.random.uniform(-(maxtheta-mintheta)/anglesteps/2,(maxtheta-mintheta)/anglesteps/2)
                phi=meanphi+np.random.uniform(-(maxphi-minphi)/anglesteps/2,(maxphi-minphi)/anglesteps/2)
                dist=meandist+np.random.uniform(-(maxdist-mindist)/diststeps/2,(maxdist-mindist)/diststeps/2)
                scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[1, 1, 1])
                camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
                intensity=10**np.random.uniform(3,3.5)
                light = pyrender.DirectionalLight(color=[1,1,1], intensity=intensity)
                scene.add(mesh, pose=np.eye(4))
                x=np.random.uniform(0,5)
                y=np.random.uniform(0,5)
                z=np.random.uniform(0,5)
                scene.add(light, pose=[[ 1,  0,  0,  x],
                                        [ 0,  1, 0, y],
                                        [ 0,  0,  1,  z],
                                        [ 0,  0,  0,  1]])
                ct=np.cos(theta)
                st=np.sin(theta)
                cp=np.cos(phi)
                sp=np.sin(phi)
                hor_rotation=np.array([[ct,0,st,0],
                                        [0,1,0,0],
                                        [-st,0,ct,0],
                                        [0,0,0,1]])
                vert_rotation=np.array([[1,0,0,0],
                                        [0,cp,-sp,0],
                                        [0,sp,cp,0],
                                        [0,0,0,1]])
                translation=np.array([[1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,dist],
                                        [0,0,0,1]])
                pose=hor_rotation.dot(vert_rotation.dot(translation))
                scene.add(camera, pose=pose)

                # render scene
                r = pyrender.OffscreenRenderer(256, 256)
                color, _ = r.render(scene)
                print(type(color))
                print(color.shape)


                voxels.apply_transform(np.linalg.pinv(pose))
                mat=pad_voxelization(voxels.matrix.astype(np.uint8),128,196,128)
                mat = mat.astype(np.uint8)
                np.savez('data_hollow/'+str(i),y=mat)
                voxels.apply_transform(pose)
                plt.imsave('data_hollow/'+str(i)+'.jpg',color)
                i=i+1
        