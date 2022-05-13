from torch.utils.data import Dataset, DataLoader
from generate_lidar import *
import os
import re
import json
import torch
from torchvision import transforms as T
import numpy as np
import cv2
import random
from matplotlib import pyplot
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

SHAPE_NPARTS = {
    "02691156": 4,
    "02773838": 2,
    "02954340": 2,
    "02958343": 4,
    "03001627": 4,
    "03261776": 3,
    "03467517": 3,
    "03624134": 2,
    "03636649": 4,
    "03642806": 2,
    "03790512": 6,
    "03797390": 2,
    "03948459": 3,
    "04099429": 3,
    "04225987": 3,
    "04379243": 3,
}

LABEL_OFFSETS = {
    "02691156": 0,
    "02773838": 4,
    "02954340": 6,
    "02958343": 8,
    "03001627": 12,
    "03261776": 16,
    "03467517": 19,
    "03624134": 22,
    "03636649": 24,
    "03642806": 28,
    "03790512": 30,
    "03797390": 36,
    "03948459": 38,
    "04099429": 41,
    "04225987": 44,
    "04379243": 47,
}

LABEL_IDX = {
    "Airplane": 0,
    "Bag": 1,
    "Cap": 2,
    "Car": 3,
    "Chair": 4,
    "Earphone": 5,
    "Guitar": 6,
    "Knife": 7,
    "Lamp": 8,
    "Laptop": 9,
    "Motorbike": 10,
    "Mug": 11,
    "Pistol": 12,
    "Rocket": 13,
    "Skateboard": 14,
    "Table": 15,
}

class SevenScenesPoints(Dataset):
    def __init__(self, root, train=True, transforms=None, max_high=25, num_samples=2048):
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.train = train
        self.max_high = max_high
        self.num_samples = num_samples

        self.image_poses = []
        self.images_path = []
        self.depths_path = []

        self._get_data()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not train:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     normalize]
                )
            else:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomCrop(224),#T.CenterCrop(224),
                     T.ToTensor(),
                     normalize]
                )

    def _get_data(self):

        if self.train:
            txt_file = self.root + '/chess_train.txt'
        else:
            txt_file = self.root + '/chess_val.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                
                # For Hacklab dataset
                imgname , p0, p1, p2, p3, p4, p5, p6 = line.split()
                depthname = imgname[:-10] + '.depth.png'
                #For Kings College
                #fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + imgname)
                self.depths_path.append(self.root + depthname)

    def __getitem__(self, index):
            """
            return the data of one image
            """
            depth_path = self.depths_path[index]
            img_path = self.images_path[index]
            img_pose = self.image_poses[index]
            img = Image.open(img_path)
            
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth_map = (depth_map).astype(np.float32)/256
            points = project_depth_to_points(depth_map, self.max_high)

            #  Sample subset of points:
            if points.shape[0] > self.num_samples:
                choice = np.random.choice(points.shape[0], self.num_samples, replace=False)
                #resample
                points = points[choice, :]
                # samples = random.sample(range(points.shape[0]), self.num_samples)
                # points = points[samples]
            
            points = points.transpose(1,0)
            points = torch.from_numpy(points).float()

            # print("Points shape:", points.shape)
            # t = T.ToTensor()
            # img = t(img)
            # print(points.shape)
            return self.transforms(img), points, img_pose
    def __len__(self):
            return len(self.images_path)



class ShapeNetDataset(Dataset):
    def __init__(self, data_path, N=1024, split=1, augment=False):
        """
        N: number of points to sample out of the total shape
        split 0: test
        split 1: train
        split 2: val
        """
        self.data_path = data_path
        assert os.path.isdir(self.data_path), ("Data Path is Not Corret: ", self.data_path)
        self.N = N
        self.augment = augment
        self.split = split

        self.class_map = self.load_class_map()
        self.file_list = self.load_files(self.split)

    def load_files(self, split):
        """ 
        split 0: test
        split 1: train
        split 2: val
        """

        split_name = [
            "shuffled_test_file_list.json",
            "shuffled_train_file_list.json",
            "shuffled_val_file_list.json",
        ]

        path = os.path.join(self.data_path, "shape_data", "train_test_split", split_name[split])
        assert os.path.isfile(path), ("Path does not exist", path)

        datas = []
        with open(path) as json_file:
            data_list = json.load(json_file)
            for data in data_list:
                d = re.split(r"/+", data)

                data = {}
                points_ = os.path.join(self.data_path, d[0], d[1], "points", d[2] + ".pts")
                label_ = os.path.join(self.data_path, d[0], d[1], "points_label", d[2] + ".seg")

                # Check if shape has enough points: 
                assert os.path.isfile(points_), ("Points file does not exist: ", points_)
                assert os.path.isfile(label_), ("Label file does not exist: ", label_)
                data["points"] = points_
                data["label"] = label_
                data["class"] = self.class_map[d[1]]
                data["folder"] = d[1]

                datas.append(data)
                
        return datas

    def load_class_map(self):
        """ 
        Create a map label -> folder_name
        """
        path = os.path.join(self.data_path, "shape_data/synsetoffset2category.txt")
        assert os.path.isfile(path), ("The file does not exist: ", path)

        f = open(path, "r")
        content = f.read()
        content_list = content.splitlines()
        f.close()

        class_map = {}
        for content in content_list:
            label, folder = re.split(r"\t+", content)
            class_map[folder] = label

        return class_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx]
        points = np.loadtxt(data["points"], delimiter=" ", dtype=np.float32)
        labels = np.loadtxt(data["label"], delimiter=" ", dtype=np.int32)
        class_name = data["class"]


        #  Sample subset of points:
        if points.shape[0] >= self.N:
            samples = random.sample(range(points.shape[0]), self.N)
            points = points[samples]
            labels = labels[samples]

        #  Correct the label offset:
        folder = data["folder"]
        offset = LABEL_OFFSETS[folder]
        labels = (labels - 1) + offset

        #  Correct the original rotation:
        points = correct_rotation(points)

        # Augment point cloud (rotation + noise)
        if self.augment:
            points = apply_augmentation(points)
        
        # Correct points shape and type
        points = points.transpose(1,0)
        points = torch.from_numpy(points)
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.LongTensor)

        class_id = np.array(LABEL_IDX[class_name])
        class_id = torch.from_numpy(class_id)

        # Hack, should find a way to skip it during the return 
        if points.shape[1] < self.N:
            points = torch.rand((3, self.N))
            labels = torch.rand((self.N))
            labels = labels.type(torch.LongTensor)

        return points, labels, class_id


def correct_rotation(points):
    th = -np.pi / 2
    c = np.cos(th)
    s = np.sin(th)
    Rx = np.array(([1, 0, 0], [0, c, -s], [0, s, c]))

    return np.matmul(points, Rx)


def apply_augmentation(points):
    # Get random rotation matrix around z:
    th = random.uniform(0, 1) * 2 * np.pi
    c = np.cos(th)
    s = np.sin(th)
    Rz = np.array(([c, -s, 0], [s, c, 0], [0, 0, 1]))

    points = np.matmul(points, Rz)

    # Change position of the points:
    noise = np.random.uniform(0, 0.2, size=points.shape)
    points += noise

    points =points.astype(np.float32)
    return points


def visualize_shape(points):
    fig = pyplot.figure()
    ax = Axes3D(fig)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    pyplot.xlabel("X axis label")
    pyplot.ylabel("Y axis label")
    pyplot.show()


if __name__ == "__main__":
    data = SevenScenesPoints('../../../Datasets/chess', num_samples=1024)
    img, points, pose = data[34]
    print("img:", img)
    print("points:", points)
    print(points.shape)
    print("pose:", pose)

    lidar = points.transpose(1, 0).numpy()
    #points = np.squeeze(points, axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar)
        
    o3d.io.write_point_cloud('test_dataset.ply', pcd)