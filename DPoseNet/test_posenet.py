import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from DataSource import *
from posenet import *
import statistics

# Function used in original PoseNet repo: https://github.com/alexgkendall/caffe-posenet
def quat_to_euler(q, is_degree=False):
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)

    return np.array([roll, pitch, yaw])


def array_dist(pred, target):
    return np.linalg.norm(pred - target, 2)


def position_dist(pred, target):
    return np.linalg.norm(pred-target, 2)


def rotation_dist(pred, target):
    pred = quat_to_euler(pred)
    target = quat_to_euler(target)

    return np.linalg.norm(pred-target, 2)


if __name__ == "__main__":

    # Resize data before using
    # ImageNet normalization params since resnet is pretrained
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                                                                                             
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PoseNet()
    model = model.to(device)

    val_dataset = DataSource("./", train=False)                                                                       
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    check_path = './check_epoch_499'
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    f = open("chess_dposenet_499.txt", "w")
    tot_pose = []
    tot_out = []
    xs = []
    ys = []
    zs = []
    outx = []
    outy = []
    outz = []
    num_poses = 0
    avg_xyz = 0
    avg_xyz_list = []
    avg_q = 0
    avg_q_list = []

    with torch.no_grad():
        for step, (rgbd, poses) in enumerate(val_loader):
            x = rgbd
            x = x.to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])                                                                                           
            poses[3] = np.array(poses[3])                                                                                           
            poses[4] = np.array(poses[4])                                                                                          
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            poses = torch.Tensor(poses).to(device)

            out = model(x)
            out = out.cpu()
            out = out.numpy()

            poses = poses.cpu()
            poses = poses.numpy()

            # Compute Error
            # Normalize quaternions
            q1 = poses[:, 3:]/ np.linalg.norm(poses[:, 3:])
            q2 = out[:, 3:] / np.linalg.norm(out[:, 3:])
            q1.squeeze(0)
            q2.squeeze(0)

            # compute error orientation
            d = abs(np.sum(np.multiply(q1,q2)))
            theta = 2 * np.arccos(d) * 180/np.pi
            avg_q += theta
            avg_q_list.append(theta)
            # compute error translartion
            dist = np.linalg.norm(poses[:, :3] - out[:,:3])
            avg_xyz += dist
            avg_xyz_list.append(dist)
            num_poses += 1
            print("Error XYZ: ")
            print(dist)
            print("Error Q")
            print(theta)

            xs.append(poses[:, 0:1])
            ys.append(poses[:, 1:2])
            zs.append(poses[:, 2:3])

            outx.append(out[:, 0:1])
            outy.append(out[:, 1:2])
            outz.append(out[:, 2:3])
            f.write("_ _ _ " + str(out[0, 0]) + " " + str(out[0, 1]) + " " + str(out[0, 2]) + " " + str(out[0, 3]) + " " + str(out[0, 4]) + " " + str(out[0, 5]) + " " + str(out[0, 6]) + "\n")
    

    avg_xyz = avg_xyz / num_poses
    avg_q = avg_q / num_poses
    print("The avg position distance is: " + str(avg_xyz))
    print("The median position distance is: " + str(statistics.median(avg_xyz_list)))
    print("The avg orientation distance is: " + str(avg_q))
    print("The median orientation distance is: " + str(statistics.median(avg_q_list)))
    f.write("The avg position distance is: " + str(avg_xyz))
    f.write("The median position distance is: " + str(statistics.median(avg_xyz_list)))
    f.write("The avg orientation distance is: " + str(avg_q)) 
    f.write("The median orientation distance is: " + str(statistics.median(avg_q_list)))
    f.close()
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the values
    ax.scatter(xs, ys, zs, c = 'b', marker='o')
    ax.scatter(outx, outy, outz, c = 'r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()
    plt.savefig('results.png')

