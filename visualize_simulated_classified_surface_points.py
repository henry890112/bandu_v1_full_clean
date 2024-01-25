import argparse

from utils.vis_util import make_color_map, make_colors
from utils import vis_util, color_util

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
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()


from utils.train_util import model_creator
from scipy.spatial.transform.rotation import Rotation as R
from utils import misc_util, surface_util
import torch
import open3d
from torch.utils.data import DataLoader
from data_generation.sim_dataset import PybulletPointcloudDataset
import json
import numpy as np

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
                               linear_search=True,
                                augment_extrinsics=False,
                                       extrinsics_noise_scale=.5,
                               depth_noise_scale=1.0)
train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=args.batch_size, drop_last=False, shuffle=False)

batch = next(iter(train_dloader))
print(batch.keys())

# if "CVAE" in models_dict['surface_classifier'].__class__.__name__:
#     if args.encoder_recon:
#         predictions = models_dict['surface_classifier'](batch)['decoder'][0].unsqueeze(1)
#         predictions_num_z = 1
#     else:
#         predictions_num_z = 1
#         predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
#                                                                      z_samples_per_sample=predictions_num_z)
# else:
#     predictions = models_dict['surface_classifier'](batch)

total_z_per_sample = 10

# create a list to store the angle between the ground truth plane and the predicted plane
angle_list = []

for round_idx in range(args.single_obj_round):
    # print the batch shape
    print("batch shape")
    print(batch['rotated_pointcloud'].shape)
    print("!!!!!!!!!!!!!!!!!!!!!!round_idx:", round_idx)

    if "CVAE" in models_dict['surface_classifier'].__class__.__name__:
        if args.encoder_recon:
            predictions = models_dict['surface_classifier'](batch)['decoder'][0].unsqueeze(1)
            predictions_num_z = 1
        else:
            predictions_num_z = 1
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                                         z_samples_per_sample=predictions_num_z)
    else:
        predictions = models_dict['surface_classifier'](batch)

    for sample_idx in range(args.batch_size):
        for z_idx in range(predictions_num_z):
            print("ln116 pred shape")
            print(predictions.shape)

            #Henry: sigmoid_threshold=.45 根本沒用到   ; topk_k=30預設 用到了
            mat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][0],
                                                                                    predictions[sample_idx][z_idx],
                                                                                    #  sigmoid_threshold=.45,)
                                                                                    topk_k=100)
        
            
            relrot = R.from_matrix(mat).as_quat()

            if "eps" in vars().keys():
                print("eps")
                print(eps[sample_idx])

            box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .5])
            arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5], scale=.1,
                                                    position=box_centroid,
                                                    # object_com=sample_pkl['position'])
                                                    object_com=np.zeros(3))  # because the object has been centered
            
            print("shape of pcd", batch['rotated_pointcloud'][sample_idx][0].shape)
            print("shape of prediction", predictions[sample_idx][z_idx].shape)

            # print(predictions[sample_idx][z_idx])
            # print(torch.sigmoid(predictions[sample_idx][z_idx]))
            # print(batch['bottom_thresholded_boolean'][sample_idx])
            #Henry make the ground truth plane
            mat_gt, plane_model_gt = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][0],
                                                                                    batch['bottom_thresholded_boolean'][sample_idx],
                                                                                    #  sigmoid_threshold=.45,)
                                                                                    topk_k=100)
            box_gt, box_centroid_gt = surface_util.gen_surface_box(plane_model_gt, ret_centroid=True, color=[0, 1, 0])
            arrow_gt = vis_util.create_arrow(plane_model_gt[:3], [0., 1., 0.], scale=.1,
                                                    position=box_centroid_gt,
                                                    # object_com=sample_pkl['position'])
                                                    object_com=np.zeros(3))
            
            # translate  box_gt y=0.1 
            box_gt.translate([0, 0, 1])
            arrow_gt.translate([0, 0, 1])
            if args.vis:
                open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                                    color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                                    vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, .5]),
                                                                                    # color=make_colors(batch['bottom_thresholded_boolean'][sample_idx][0][:, 0],
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
                
                # #Henry: testtest
                open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))), 
                                                    box,
                                                    arrow,
                                                    open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                
                # # open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, .5]),
                # #                                                                     # color=make_colors(batch['bottom_thresholded_boolean'][sample_idx][0][:, 0],
                # #                                                                     # background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                # #                                                                     color=make_color_map(torch.sigmoid(predictions[sample_idx][z_idx].squeeze(-1))))])
                # #Henry: yellow is ground truth？?？?？
                # open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                #                                                                     color=make_colors(batch['bottom_thresholded_boolean'][sample_idx],
                #                                                                     background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                #                                     box_gt,
                #                                     arrow_gt,
                #                                     open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                
            #already the unit vector
            print("predict arror", plane_model[:3])
            print("gt arrow", plane_model_gt[:3])      
            
            # calculate the angle between two vectors
            angle = np.arccos(np.clip(np.dot(plane_model[:3], plane_model_gt[:3]), -1.0, 1.0))
            # change the radian to degree
            angle = angle * 180 / np.pi
            if angle > 140:
                angle = 180 - angle
            print("angle between two vectors", angle)
            angle_list.append(angle)

# calculate the average angle
angle_list = np.array(angle_list)
print("average angle", np.mean(angle_list))

# use the last directory name in train_dset_path to be the subdir
subdir = args.train_dset_path.split("/")[-1]

txt_path = "/home/docker/bandu_v1_full_clean/txt_file/"
# 畫出angle_list的折線圖
import matplotlib.pyplot as plt
print("angle_list", angle_list)
plt.plot(angle_list)
# control the range of y-axis
plt.ylim(0, 180)
plt.ylabel('angle')
plt.xlabel('sample')
# use the subdirectory name as the title
plt.title(subdir)
# show the numbers that angle is smaller than 30 and print in the plot on the top right
rate = np.sum(angle_list < 30) / len(angle_list)
rate *= 100
plt.text(0, 170, "rate: " + str(rate) + "%", fontsize=10)
# save the plt.show() plot in the txt_file directory
plt.savefig(txt_path + subdir + ".png")
plt.show()



import os
# clean the old txt file and write the new angle_list into txt file(file name is subdir)
# check the txt file is exist or not
if os.path.exists(txt_path + subdir + ".txt"):
    os.remove(txt_path + subdir + ".txt")
f = open(txt_path + subdir + ".txt", "w")
f.write(str(angle_list))
f.close()









        