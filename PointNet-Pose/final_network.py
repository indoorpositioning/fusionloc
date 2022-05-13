import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from attention import AttentionBlock
from network import *

class RGBPointLoc(nn.Module):
    def __init__(self, device):
        """
        k: number of classes which a the input (shape) can be classified into
        """
        super(RGBPointLoc, self).__init__()
        self.device = device

        # resnet as feature extractor
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.5)
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # Parameters of newly constructed modules have requires_grad=True by default
        self.rgb_fc_features = 2048#1024
        self.points_fc_features = 0#1024
        self.final_fc_features = 1024
        #self.loc_features = 512
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.rgb_fc_features)
        # self.TNet3 = TNet3(self.device)
        # self.TNet64 = TNet64(self.device)

        self.mlp1 = mlp(3, 64, batchnorm=False)
        self.mlp2 = mlp(64, 64, batchnorm=False)
        self.mlp3 = mlp(64, 128, batchnorm=False)
        self.mlp4 = mlp(128, 1024, batchnorm=False)

        self.att = AttentionBlock(self.rgb_fc_features)

        self.fc1 = torch.nn.Linear(self.rgb_fc_features + self.points_fc_features, self.final_fc_features)
        # self.fc2 = torch.nn.Linear(512, 128)
        # self.fc3 = torch.nn.Linear(128, 128)
        #self.fc1 = torch.nn.Linear(3072, 512)
        
        self.fc_pose_xyz = torch.nn.Linear(self.final_fc_features, 3)
        self.fc_pose_wpqr = torch.nn.Linear(self.final_fc_features, 4)

    def forward(self, rgb, points):
        x = self.resnet(rgb)
        # attention on rgb
        x = self.att(x)

        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(x)

        # final output
        out_xyz = self.fc_pose_xyz(x)
        out_wpqr = self.fc_pose_wpqr(x)
        out = torch.cat((out_xyz, out_wpqr), dim=1)
        
        return out

    def forward2(self, rgb, points):
        #  input transform:
        # point_out_ = point_out.clone()
        # T3 = self.TNet3(point_out_)
        # point_out = torch.matmul(T3, point_out)

        # rgb images
        rgb_out = self.resnet(rgb)
        # attention on rgb
        rgb_out = self.att(rgb_out)
        # x = F.relu(x)

        # point cloud
        #  mlp (64,64):
        point_out = self.mlp1(points)
        point_out = self.mlp2(point_out)
        
        # feature transform:
        # point_out_ = point_out.clone()
        # T64 = self.TNet64(point_out_)
        # point_out = torch.matmul(T64, point_out)
        
        #  mlp (64,128,1024):
        point_out = self.mlp3(point_out)
        point_out = self.mlp4(point_out)

        

        # max pool
        point_out = torch.max(point_out, 2, keepdim=True)[0]
        
        # point pose layers
        point_out = torch.flatten(point_out, start_dim=1)

        # concatenate rgb + lidar = 2048 + 1024 = 3072
        x = torch.cat((rgb_out, point_out), dim=1)


        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        x = F.relu(x)

        # final output
        out_xyz = self.fc_pose_xyz(x)
        out_wpqr = self.fc_pose_wpqr(x)
        out = torch.cat((out_xyz, out_wpqr), dim=1)
        
        return out
