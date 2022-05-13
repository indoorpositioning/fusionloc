import argparse
import os

import numpy as np
# import scipy.misc as ssc
import matplotlib.pyplot as plt
import cv2
import open3d as o3d


# =========================== 
# ------- 3d to 3d ---------- 
# =========================== 
# def project_ref_to_velo(self, pts_3d_ref):
#     pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
#     return np.dot(pts_3d_ref, np.transpose(self.C2V))

# def project_rect_to_ref(self, pts_3d_rect):
#     ''' Input and Output are nx3 points '''
#     return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))

# def project_rect_to_velo(self, pts_3d_rect):
#     ''' Input: nx3 points in rect camera coord.
#         Output: nx3 points in velodyne coord.
#     '''
#     pts_3d_ref = project_rect_to_ref(pts_3d_rect)
#     return project_ref_to_velo(pts_3d_ref)

# =========================== 
# ------- 2d to 3d ---------- 
# =========================== 
def project_image_to_rect(uv_depth):
    ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    # We don't plan on working with stereo so set baseline to 0
    b_x, b_y = 0, 0
    f_u, f_v = 585, 585
    c_u, c_v = 320, 240
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    # print(x.max(), y.max(), uv_depth[:, 2].max())
    return pts_3d_rect

# def project_image_to_velo(uv_depth):
#     pts_3d_rect = project_image_to_rect(uv_depth)
#     return pts_3d_rect

# def project_disp_to_points(calib, disp, max_high):
#     disp[disp < 0] = 0
#     baseline = 0.54
#     mask = disp > 0
#     depth = calib.f_u * baseline / (disp + 1. - mask)
#     rows, cols = depth.shape
#     c, r = np.meshgrid(np.arange(cols), np.arange(rows))
#     points = np.stack([c, r, depth])
#     points = points.reshape((3, -1))
#     points = points.T
#     points = points[mask.reshape(-1)]
#     cloud = calib.project_image_to_velo(points)
#     valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
#     return cloud[valid]

def project_depth_to_points(depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = project_image_to_rect(points)#calib.project_image_to_velo(points)
    valid = (cloud[:, 2] <= max_high)#((cloud[:, 0]) >= 0)# & (cloud[:, 2] < max_high)
    #print(cloud[:, 1].min())
    return cloud[valid]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Libar')
    # parser.add_argument('--calib_dir', type=str,
    #                     default='~/Kitti/object/training/calib')
    parser.add_argument('--depth_dir', type=str,
                        default='../depth_data')
    parser.add_argument('--rgb_dir', type=str,
                        default='../rgb_data')
    parser.add_argument('--save_dir', type=str,
                        default='../projected_maps')
    parser.add_argument('--max_high', type=int, default=25)
    # parser.add_argument('--is_depth', action='store_true')

    args = parser.parse_args()

    assert os.path.isdir(args.depth_dir)
    # assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.depth_dir) if x[-9:] == 'depth.png' or x[-3:] == 'npy']
    rgbs = [x for x in os.listdir(args.rgb_dir) if x[-9:] == 'color.png']
    disps = sorted(disps)
    print(len(disps))
    for fn in disps:
        predix = fn[:-10]
        print(predix, fn[-9:])
        # calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        # calib = kitti_util.Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        if fn[-9:] == 'depth.png':
            depth_map = cv2.imread(args.depth_dir + '/' + fn, cv2.IMREAD_ANYDEPTH)#ssc.imread(args.depth_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            depth_map = np.load(args.depth_dir + '/' + fn)
        else:
            assert False
        bgr_map = cv2.imread(args.rgb_dir + '/' + predix + '.color.png')
        bgr_map = cv2.resize(bgr_map, depth_map.shape, interpolation = cv2.INTER_AREA)
        rgb_map = cv2.cvtColor(bgr_map, cv2.COLOR_BGR2RGB)
        rgb_map = rgb_map.reshape((3, -1))
        rgb_map = rgb_map.T
        # if not args.is_depth:
        #     disp_map = (disp_map*256).astype(np.uint16)/256.
        #     lidar = project_disp_to_points(calib, disp_map, args.max_high)
        # else:
        depth_map = (depth_map).astype(np.float32)/256.
        lidar = project_depth_to_points(depth_map, args.max_high)
        print(lidar[:])
        ax = plt.axes(projection='3d')
        ax.scatter(lidar[:,0], lidar[:,1], lidar[:,2], c = 'blue', s=0.01)
        #plt.show()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar)

        
        # pcd.colors = o3d.utility.Vector3dVector(rgb_map.astype(np.float) / 255.0)
        
        o3d.io.write_point_cloud('{}/{}.ply'.format(args.save_dir, predix), pcd)
        
        # pad 1 in the indensity dimension
        #lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        # lidar = np.dstack((lidar, rgb_map))
        # lidar = lidar.astype(np.float32)
        
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
