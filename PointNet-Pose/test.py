import numpy as np
import matplotlib.pyplot as plt                                                                                                       #import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import SevenScenesPoints
from network_v2 import PointNetPose
from network import PointNetPose_v1
from final_network import RGBPointLoc
from loss import PoseLoss
# from utils import qr

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
    model = PointNetPose_v1(device)#RGBPointLoc(device)
    model = model.to(device)

    val_dataset = SevenScenesPoints("../Datasets/chess", train=False, num_samples=1024)                                                                       
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    check_path = './models/check_epoch_230.pt'
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    f = open("pointnetloc_revisit_epoch230.txt", "w")#"chess_plidar_150_notrans_512_attention1024.txt", "w")
    tot_pose = []
    tot_out = []
    xs = []
    ys = []
    zs = []
    outx = []
    outy = []
    outz = []
    distances = []
    angles = []
    num_poses = 0
    avg_xyz = 0
    avg_q = 0

    with torch.no_grad():
        for step, (rgbs, points, poses) in enumerate(val_loader):
            points = points.to(device)
            rgbs = rgbs.to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])                                                                                           
            poses[3] = np.array(poses[3])                                                                                           
            poses[4] = np.array(poses[4])                                                                                          
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            poses = torch.Tensor(poses).to(device)

            out = model(points)
            out = out.cpu()
            out = out.numpy()

            poses = poses.cpu()
            poses = poses.numpy()
            # #print(out.shape)

            ##ATLOC
            # # normalize the predicted quaternions
            # q = [qexp(p[3:]) for p in output]
            # out = np.hstack((out[:, :3], np.asarray(q)))
            # q = [qexp(p[3:]) for p in target]
            # target = np.hstack((target[:, :3], np.asarray(q)))

            # # take the middle prediction
            # pred_poses[idx, :] = output[len(output) / 2]
            # targ_poses[idx, :] = target[len(target) / 2]

            ####POSENET

            # pose_q = poses[:, 3:]
            # pose_x = poses[:, :3]
            # predicted_q = out[:, 3:]
            # predicted_x = out[:, :3]

            # # pose_q = np.repeat([pose_q],sample_size, axis=0)
            # # pose_x = np.repeat([pose_x],sample_size, axis=0)
            # # predicted_q = np.squeeze(predicted_q)
            # # predicted_x = np.squeeze(predicted_x)
            # # predictions = np.concatenate((predicted_x,predicted_q),axis=1)

            # #Compute Individual Sample Error
            # q1 = pose_q / np.linalg.norm(pose_q,axis=1)[:,None]
            # q2 = predicted_q / np.linalg.norm(predicted_q,axis=1)[:,None]
            # d = abs(np.sum(np.multiply(q1,q2),axis=1))
            # theta_individual = 2 * np.arccos(d) * 180/np.pi
            # error_x_individual = np.linalg.norm(pose_x-predicted_x,axis=1)

            # #Total error
            # predicted_x_mean = np.mean(predicted_x,axis=0)
            # predicted_q_mean = np.mean(predicted_q,axis=0)
            # pose_x = np.mean(pose_x,axis=0)
            # pose_q = np.mean(pose_q,axis=0)

            # q1 = pose_q / np.linalg.norm(pose_q)
            # q2 = predicted_q_mean / np.linalg.norm(predicted_q_mean)
            # d = abs(np.sum(np.multiply(q1,q2)))
            # theta = 2 * np.arccos(d) * 180/np.pi
            # error_xyz = pose_x-predicted_x_mean
            # error_x = np.linalg.norm(error_xyz)

            # avg_q += theta
            # angles.append(theta)
            # avg_xyz += error_x
            # num_poses += 1
            # distances.append(error_x)

            #Variance
            # covariance_x = np.cov(predicted_x,rowvar=0)
            # uncertainty_x = covariance_x[0,0] + covariance_x[1,1] + covariance_x[2,2]

            # covariance_q = np.cov(predicted_q,rowvar=0)
            # uncertainty_q = math.sqrt(covariance_q[0,0] + covariance_q[1,1] + covariance_q[2,2] + covariance_q[3,3])

            # results[count,:] = [error_x, math.sqrt(uncertainty_x), theta, math.sqrt(uncertainty_q)]

            # count += 1


            # # print 'Iteration:  ', count
            # # print 'Error XYZ (m):  ', error_x
            # # print 'Uncertainty XYZ:   ', uncertainty_x
            # # print 'Error Q (degrees):  ', theta
            # # print 'Uncertainty Q:   ', uncertainty_q
            
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
            angles.append(theta)
            # compute error translartion
            dist = np.linalg.norm(poses[:, :3] - out[:,:3])
            avg_xyz += dist
            num_poses += 1
            distances.append(dist)
            # print("Error XYZ: ")
            # print(dist)
            # print("Error Q")
            #print(theta)

            xs.append(poses[:, 0:1])
            ys.append(poses[:, 1:2])
            zs.append(poses[:, 2:3])

            outx.append(out[:, 0:1])
            outy.append(out[:, 1:2])
            outz.append(out[:, 2:3])
            f.write("_ _ _ " + str(out[0, 0]) + " " + str(out[0, 1]) + " " + str(out[0, 2]) + " " + str(out[0, 3]) + " " + str(out[0, 4]) + " " + str(out[0, 5]) + " " + str(out[0, 6]) + "\n")
    

    assert num_poses == len(distances) and num_poses == len(angles), "Length must match"

    med_xyz = np.median(distances)
    med_q = np.median(angles)

    avg_xyz = avg_xyz / num_poses
    avg_q = avg_q / num_poses
    print("The avg position distance is: " + str(avg_xyz))
    print("The avg orientation distance is: " + str(avg_q))
    print("The median position distance is: " + str(med_xyz))
    print("The median orientation distance is: " + str(med_q))

    f.write("The avg position distance is: " + str(avg_xyz))
    f.write("The avg orientation distance is: " + str(avg_q)) 
    f.write("The median position distance is: " + str(med_xyz))
    f.write("The median orientation distance is: " + str(med_q)) 
    f.close()
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Generate the values
    # xs = tot_pose[:, 0:1]
    # ys = tot_pose[:, 1:2]
    # zs = tot_pose[:, 2:3]
    # Plot the values
    ax.scatter(xs, ys, zs, c = 'b', marker='o')
    ax.scatter(outx, outy, outz, c = 'r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
