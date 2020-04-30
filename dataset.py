import numpy as np
import os
import png
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image

class PeopleDataset(Dataset):
    def __init__(self,flag, dataDir='./data/', data_range=(0, 8)):
        assert(flag in ['train', 'eval', 'test',''])
        print("load "+ flag+" dataset start")
        print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        self.flag=flag
        self.dataDir=dataDir
        for i in range(data_range[0], data_range[1]):
            img = Image.open(os.path.join(dataDir,flag,'%d.jpg' % i))
            label=np.load(os.path.join(dataDir,flag,'%d.npz' % i))['y']
            # Normalize input image
            img = np.asarray(img).astype("f").transpose(2, 0, 1)/128.0-1.0
            if(label.shape[2]==128):
                self.dataset.append(i)
        print("load dataset done")
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataDir,self.flag,'%d.jpg' % self.dataset[index]))
        label=np.load(os.path.join(self.dataDir,self.flag,'%d.npz' % self.dataset[index]))['y']
        # Normalize input image
        img = np.asarray(img).astype("f").transpose(2, 0, 1)/128.0-1.0
        label = torch.FloatTensor(label)

        return torch.FloatTensor(img), torch.FloatTensor(label)

