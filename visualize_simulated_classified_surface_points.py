import argparse

from utils.vis_util import make_color_map, make_colors, make_colors_topk
from utils import vis_util, color_util
from utils.train_util import model_creator
from scipy.spatial.transform.rotation import Rotation as R
from utils import misc_util, surface_util
import torch
import open3d
from torch.utils.data import DataLoader
from data_generation.sim_dataset import PybulletPointcloudDataset
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('resume_pkl', type=str, help="Checkpoint to resume from")
parser.add_argument('train_dset_path', type=str)

parser.add_argument('--device_id', default=0)
parser.add_argument('--num_points', type=int, default=150, help="Num points for FPS sampling")
parser.add_argument('--num_fps_samples', type=int, default=1, help="Num samples for FPS sampling")
parser.add_argument('--resume_initconfig', type=str, help="Initconfig to resume from")
parser.add_argument('--stats_json', type=str)

#Henry：更改要test的數量
parser.add_argument('--batch_size', type=int, default=1) 
parser.add_argument('--encoder_recon', action='store_true', help="Use encoder for reconstruction")
parser.add_argument('--val_dset_path', type=str)
parser.add_argument('--center_fps_pc', action='store_true', help='Center FPS')

#Henry add to vali default=0date the model on single object
parser.add_argument('--single_obj_round', type=int, default=1)
parser.add_argument('--vis', default=False, action='store_true')
parser.add_argument('--use_cluster', default=True, action='store_true')
parser.add_argument('--get_toward_vector', default=False, action='store_true')
parser.add_argument('--topk_k', type=int, default=100)
parser.add_argument('--validate_parm_mode', default=False, action='store_true')

args = parser.parse_args()

torch.set_printoptions(edgeitems=12)
config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)
# load model
models_dict = model_creator(config=config,
                            device_id=args.device_id)

device = torch.device("cuda:0")
sd = torch.load(args.resume_pkl, map_location=device)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)


models_dict['surface_classifier'].load_state_dict(sd['model'])
models_dict['surface_classifier'].eval()

train_dset = PybulletPointcloudDataset(args.train_dset_path,
                               stats_dic=stats_dic,
                               center_fps_pc=args.center_fps_pc,
                                rot_aug=None,
                               shear_aug=None,
                               scale_aug=None,
                               threshold_frac=.02,
                               max_frac_threshold=.1,
            #Henry20240306: sigmoid_threshold=.45 根本沒用到   ; topk_k=30預設 用到了
                               linear_search=True,
                                augment_extrinsics=False,
                                       extrinsics_noise_scale=.5,
                               depth_noise_scale=1.0)
train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=args.batch_size, drop_last=False, shuffle=False)

batch = next(iter(train_dloader))
print(batch.keys())
mean_angle = 180
check_round = 0

while(mean_angle > 15):
    check_round += 1
    angle_list = []
    for round_idx in range(args.single_obj_round):
        # print the batch shape
        # print("batch shape")
        # print(batch['rotated_pointcloud'].shape)
        print("round_idx:", round_idx)

        if "CVAE" in models_dict['surface_classifier'].__class__.__name__:
            if args.encoder_recon:
                print("encoder_recon")
                predictions = models_dict['surface_classifier'](batch)['decoder'][0].unsqueeze(1)
                predictions_num_z = 1
            else:
                predictions_num_z = 1
                predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                                            z_samples_per_sample=predictions_num_z)
        else:
            predictions = models_dict['surface_classifier'](batch)

        for sample_idx in range(args.batch_size):
            for z_idx in range(predictions_num_z):

                #Henry20240306: sigmoid_threshold=.45 根本沒用到   ; topk_k=30預設 用到了
                #Henry 更改topK的數量 及 get_relative_rotation_from_binary_logits中的min_num_points
                mat, plane_model, predicted_cluster_pcd = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][0],
                                                                                        predictions[sample_idx][z_idx],
                                                                                        #  sigmoid_threshold=.45,)
                                                                                        min_num_points=3,  # 因為要三個點才有平面
                                                                                        topk_k=args.topk_k,
                                                                                        use_cluster=args.use_cluster) 
            
                
                relrot = R.from_matrix(mat).as_quat()

                if "eps" in vars().keys():
                    print("eps")
                    print(eps[sample_idx])

                box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .5])

                #Henry create_arrow中可以print rot_mat    # print(rot_mat)
                arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5], scale=.1,
                                                        position=box_centroid,
                                                        # object_com=sample_pkl['position'])
                                                        object_com=np.zeros(3))  # because the object has been centered
                
                #Henry make the ground truth plane
                mat_gt, plane_model_gt, _ = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][0],
                                                                                        batch['bottom_thresholded_boolean'][sample_idx],
                                                                                        #  sigmoid_threshold=.45,)
                                                                                        use_cluster=args.use_cluster)
                
                box_gt, box_centroid_gt = surface_util.gen_surface_box(plane_model_gt, ret_centroid=True, color=[0, 1, 0])
                arrow_gt = vis_util.create_arrow(plane_model_gt[:3], [0., 1., 0.], scale=.1,
                                                        position=box_centroid_gt,
                                                        # object_com=sample_pkl['position'])
                                                        object_com=np.zeros(3))
                
                # translate  box_gt y=0.1 
                box_gt.translate([0, 0, 1])
                arrow_gt.translate([0, 0, 1])

                # calculate the angle between two vectors
                angle = np.arccos(np.clip(np.dot(plane_model[:3], plane_model_gt[:3]), -1.0, 1.0))
                angle = angle * 180 / np.pi

                if angle > 140:
                    angle = 180 - angle

                print("angle between two vectors", angle)
                angle_list.append(angle)

                if args.vis:
                    open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                                        color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                                        vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, .5]),
                                                                                        # color=make_colors(batch[https://kktix.com/events/w8a2gthy01/registrations/new'bottom_thresholded_boolean'][sample_idx][0][:, 0],
                                                                                        # background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                                                                                        color=make_color_map(torch.sigmoid(predictions[sample_idx][z_idx].squeeze(-1)))),
                                                        vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, 1]),
                                                                                        color=make_colors(batch['bottom_thresholded_boolean'][sample_idx],
                                                                                        background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                                            
                                                        box,
                                                        arrow,
                                                        box_gt,
                                                        arrow_gt,
                                                        open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
        
                if args.validate_parm_mode:
                    if angle > 30:
                        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, .5]),
                                                                                            # color=make_colors(batch[https://kktix.com/events/w8a2gthy01/registrations/new'bottom_thresholded_boolean'][sample_idx][0][:, 0],
                                                                                            # background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                                                                                            color=make_color_map(torch.sigmoid(predictions[sample_idx][z_idx].squeeze(-1)))),
                                                            vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],color=make_colors_topk(torch.sigmoid(predictions[sample_idx][z_idx]), k=args.topk_k)), 
                                                            predicted_cluster_pcd,
                                                            box,
                                                            arrow,
                                                            open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                

                if args.get_toward_vector:  
                    center = torch.mean(batch['rotated_pointcloud'][sample_idx][0], axis=0)

                    # 獲取arrow向量
                    use_gt = False
                    print("predict arror", plane_model[:3])
                    print("gt arrow", plane_model_gt[:3])

                    if use_gt:
                        n = plane_model_gt[:3]
                        arrow = vis_util.create_arrow(plane_model_gt[:3], [0., 0., 1.], scale=.1,
                                                        position=center,
                                                        object_com=np.zeros(3)) 
                    else:
                        n = plane_model[:3]
                        arrow = vis_util.create_arrow(plane_model[:3], [0., 0., 1.], scale=.1,
                                                        position=center,
                                                        object_com=np.zeros(3))
                    
                    toward_point = batch['rotated_pointcloud'][sample_idx][0]
                    # 法向量 plane model是從ransac中獲得的(a, b, c, d) -> ax+by+cz+d=0
                    norm = np.linalg.norm(n)

                    # 計算投影
                    projections = toward_point - np.outer(np.dot(toward_point, n) / norm**2, n)

                    # 創建一個新的點雲對象來存儲投影的點
                    pcd_projection = open3d.geometry.PointCloud()
                    pcd_projection.points = open3d.utility.Vector3dVector(projections)
                    pcd_projection.paint_uniform_color([0, 0, 1])

                    # 對projection利用pca
                    print('projections', projections.shape)
                    centroid = torch.mean(projections, axis=0)
                    points_centered = projections - centroid
                    cov_matrix = np.cov(points_centered, rowvar=False)
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

                    # 特徵值從大到小排序，並排序特徵向量
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvectors = eigenvectors[:, idx]

                    # 主成分
                    principal_axis = eigenvectors[:, 0]
                    second_principal_axis = eigenvectors[:, 1]
                    third_principal_axis = eigenvectors[:, 2]
                    print("principal_axis", principal_axis)

                    # vis eigenvector arrow
                    arrow_toward = vis_util.create_arrow(second_principal_axis, [0., 1., 0.], scale=.1,
                                                        position=centroid,
                                                        object_com=np.zeros(3))
                    if args.vis:
                        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                                            pcd_projection, 
                                                            open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                        
                        open3d.visualization.draw_geometries([pcd_projection, 
                                                                arrow_toward,
                                                                arrow])
                        
                        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                                                arrow_toward,
                                                                arrow])

    # calculate the average angle
    angle_list = np.array(angle_list)
    mean_angle = np.mean(angle_list)
    print("average angle", mean_angle)
    print(f"=========check_round: {check_round}==============")

# use the last directory name in train_dset_path to be the subdir
subdir = args.train_dset_path.split("/")[-1]



txt_path = "/home/docker/bandu_v1_full_clean/txt_file/"
# 畫出angle_list的折線圖
print("angle_list", angle_list)
plt.plot(angle_list)
# control the range of y-axis
plt.ylim(0, 180)
plt.ylabel('angle')
plt.xlabel('sample')
# use the subdirectory name as the title
plt.title(subdir)
# show the numbers that angle is smaller than 30 and print in the plot on the top right
rate_20 = np.sum(angle_list < 20) / len(angle_list)
rate_20 *= 100
rate_15 = np.sum(angle_list < 15) / len(angle_list)
rate_15 *= 100
rate_10 = np.sum(angle_list < 10) / len(angle_list)
rate_10 *= 100
plt.text(0, 170, "rate ( < 20): " + str(round(rate_20, 2)) + "%", fontsize=10)
plt.text(0, 160, "rate ( < 15): " + str(round(rate_15, 2)) + "%", fontsize=10)
plt.text(0, 150, "rate ( < 10): " + str(round(rate_10, 2)) + "%", fontsize=10)
plt.text(0, 140, "average angle: " + str(round(np.mean(angle_list), 2)), fontsize=10)
plt.text(0, 130, "mean: " + str(round(np.mean(angle_list), 2)) , fontsize=10)
plt.text(0, 120, " std: " + str(round(np.std(angle_list), 2)), fontsize=10)
# save the plt.show() plot in the txt_file directory
plt.savefig(txt_path + subdir + ".png")
plt.show()


# clean the old txt file and write the new angle_list into txt file(file name is subdir)
# check the txt file is exist or not
if os.path.exists(txt_path + subdir + ".txt"):
    os.remove(txt_path + subdir + ".txt")
f = open(txt_path + subdir + ".txt", "w")
f.write(str(angle_list))
f.close()

# create a new column with name subdir in the old csv file to record the angle list
csv_path = "/home/docker/bandu_v1_full_clean/csv_file/"
# check the csv file is exist or not
if os.path.exists(csv_path + "angle_list.csv"):
    df = pd.read_csv(csv_path + "angle_list.csv")
    df[subdir] = angle_list
    df.to_csv(csv_path + "angle_list.csv", index=False)
else:
    df = pd.DataFrame(angle_list, columns=[subdir])
    df.to_csv(csv_path + "angle_list.csv", index=False)

print(f"=========check_round: {check_round}==============")