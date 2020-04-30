import torch
import builder
import torch.nn as nn

#Keypoint Extractor relies upon a pose estimator which can be found at https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation
class KPE(nn.Module):
    def __init__(self,upsample=4):
        super(KPE,self).__init__()
        Builder = builder.Builder()
        self.Model = Builder.Model()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample)
    def forward(self,x):
        l=self.Model(x)
        l=l[1]
        maps=self.upsample(l)
        return maps
