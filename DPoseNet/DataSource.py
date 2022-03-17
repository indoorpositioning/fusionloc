import os
import os.path
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Transforms applied when concateneated cannot apply random crop sepretaly
class DataSource(data.Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.train = train

        self.image_poses = []
        self.depths_path = []

        self._get_data()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not train:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.CenterCrop(224),
                     normalize]
                )
            else:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomCrop(224),
                     normalize]
                )

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'chess_train.txt'
        else:
            txt_file = self.root + 'chess_val.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                
                # For 7 scenes dataset
                depthname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.depths_path.append(self.root + '../../7Scenes/chess' + depthname)

    def __getitem__(self, index):
        """
        return the data of one image
        """
        depth_path = self.depths_path[index]
        img_pose = self.image_poses[index]
        
        # encode with OPENCV EXOENSIVE ON CPU
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        norm_depth = cv2.normalize(depth, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        bgr_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        rgb_depth = cv2.cvtColor(bgr_depth, cv2.COLOR_BGR2RGB) # alternatively to PIL use atype(uint8)
        rgb_depth = Image.fromarray(rgb_depth)
        
        t = T.ToTensor()
        rgb_depth = t(rgb_depth)

        return self.transforms(rgb_depth), img_pose

    def __len__(self):
        return len(self.depths_path)
