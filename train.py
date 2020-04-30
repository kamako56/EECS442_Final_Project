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
from Model import Version2
from dataset import PeopleDataset




def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = []
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        i=0
        outputs=torch.stack(outputs,axis=0)
        labels=torch.stack((labels,labels,labels,labels),axis=0)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()
        running_loss.append( loss.item())
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, np.mean(running_loss), end-start))
    return np.mean(running_loss)

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[3]
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size=4
    train_data = PeopleDataset(flag='', data_range=(0,2000))
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=4)
    val_data = PeopleDataset(flag='', data_range=(2000,2375))
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True,num_workers=4)

    name = 'experiment5'
    net = Version2().to(device)
    epoch_to_load = -1
    # net.load_state_dict(torch.load('./models/model_{}_epoch{}.pth'.format(name,epoch_to_load)))
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(net.parameters(), 1e-4)
    train_loss=[]
    val_loss=[]
    print('\nStart training')
    for epoch in range(epoch_to_load+1,80): #TODO decide epochs
        if epoch==40:
            optimizer = torch.optim.RMSprop(net.parameters(), 1e-5)
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train_loss.append(train(train_loader, net, criterion, optimizer, device, epoch+1))
        val_loss.append(test(val_loader, net, criterion, device))
        torch.save(net.state_dict(), './models/model_{}_epoch{}.pth'.format(name,epoch))


    plt.plot(train_loss, 'o', label='train')
    plt.plot(val_loss, 'o', label='val')
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()