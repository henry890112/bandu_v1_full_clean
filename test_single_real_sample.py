import torch
from utils import misc_util, train_util, surface_util, vis_util
from utils.vis_util import make_color_map, make_colors, make_colors_topk
import open3d
import argparse
import numpy as np
import json
import open3d as o3d
from utils import color_util
from scipy.spatial.transform.rotation import Rotation as R
import copy
import os
from PIL import Image
import matplotlib.pyplot as plt
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('sc_checkpoint', help="Model checkpoint: surface_classifier")
parser.add_argument('--sample_path')
parser.add_argument('--stats_json')
parser.add_argument('--device_id', default=0)
parser.add_argument('--vis', default=True, action='store_true')
parser.add_argument('--topk_k', type=int, default=100)
parser.add_argument('--get_toward_vector', default=True, action='store_true')
parser.add_argument('--fps_pc', default=False, action='store_true', help='Center FPS')



args = parser.parse_args()
config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)
models_dict = train_util.model_creator(config=config,
                            device_id=args.device_id)

sd = torch.load(args.sc_checkpoint, map_location="cpu")
models_dict['surface_classifier'].load_state_dict(sd['model'])
predictions_num_z = 1
batch = dict()
sample_pkl = dict()

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            pc = torch.from_numpy(pc).cuda()[None].float()
            new_xyz = (
                gather_operation(
                    pc.transpose(1, 2).contiguous(), furthest_point_sample(pc[..., :3].contiguous(), npoints)
                )
                .contiguous()
                )
            pc = new_xyz[0].T.detach().cpu().numpy()

        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False
            )
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


# load the npy file
parent_directory = os.path.join(os.path.dirname(__file__))
multiview_path = os.path.join(parent_directory, 'my_test_data/pc_segments_combine_noise.npy')
multiview_pointcloud = np.load(multiview_path)
sample_pkl['points'] = multiview_pointcloud

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(multiview_pointcloud)
pcd.paint_uniform_color([0, 0, 0])
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
# if args.vis:
#     o3d.visualization.draw_geometries([pcd, origin_frame])
# Henry center
center = np.mean(multiview_pointcloud, axis=0)
pcd.translate(-center)
# visualize pointcloud
# if args.vis:
#     open3d.visualization.draw_geometries([pcd, origin_frame])

points = np.asarray(pcd.points)

# 使用最遠點採樣算法來下採樣點雲到 2048 個點
point_state = regularize_pc_point_count(points, 2048, use_farthest_point=args.fps_pc)
downsampled_pcd = o3d.geometry.PointCloud()
downsampled_pcd.points = o3d.utility.Vector3dVector(point_state)
downsampled_pcd.paint_uniform_color([0, 0, 0])

print('=====downsampled_pcd=====: ', np.array(downsampled_pcd.points).shape)
batch['rotated_pointcloud'] = torch.from_numpy(np.array(downsampled_pcd.points)).unsqueeze(0).unsqueeze(0)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)
batch['rotated_pointcloud_mean'] = torch.Tensor(stats_dic['rotated_pointcloud_mean'])
batch['rotated_pointcloud_var'] = torch.as_tensor(stats_dic['rotated_pointcloud_var'])
models_dict['surface_classifier'].eval()
predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                             z_samples_per_sample=predictions_num_z)

# predictions[0][0] 為預測的confidence (N, 1)
print('=====predictions=====:', predictions[0][0].shape)

rotmat, plane_model, predicted_cluster_pcd = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][0][0],
                                                                            predictions[0][0],
                                                                            topk_k=args.topk_k)

geoms_to_draw = []
box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .1])
arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                             position=box_centroid,
                                             object_com=np.zeros(3)) # because the object has been centered

#total visualize
if args.vis:
    open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                              color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                              vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0]+ torch.tensor([0, 0, .5]),
                              color=vis_util.make_color_map(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                                predicted_cluster_pcd,
                                box,
                                arrow])
sample_idx = 0
z_idx = 0
if args.get_toward_vector:  
    center = torch.mean(batch['rotated_pointcloud'][sample_idx][0], axis=0)

    # 獲取arrow向量
    use_gt = False
    print("predict arror", plane_model[:3])

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
                                        position=center,
                                        object_com=np.zeros(3))
    if args.vis:
        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                            pcd_projection])
        
        open3d.visualization.draw_geometries([pcd_projection, 
                                                arrow_toward,
                                                arrow])
        
        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                                arrow_toward,
                                                arrow])
# below creates the cp binary mask from topk
# -> [num_points]
contact_points_binary_mask = torch.zeros_like(predictions[0][0].squeeze())
contact_points_binary_mask[torch.topk(predictions[0][0].squeeze(), args.topk_k, largest=False)[-1]] = 1
surface_points_binary_mask = 1 - contact_points_binary_mask

# how to get colors
zeroed_out_colors = (torch.from_numpy(np.array(downsampled_pcd.colors)).to(surface_points_binary_mask.device) * surface_points_binary_mask.unsqueeze(-1))
original_rgb_with_red_contact_points = zeroed_out_colors + \
                                       contact_points_binary_mask.unsqueeze(-1) * torch.Tensor([1, 0, 0]).unsqueeze(0).expand(surface_points_binary_mask.shape[0], -1).to(contact_points_binary_mask.device)

object_realsense_pcd = vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                                    color=original_rgb_with_red_contact_points.data.cpu().numpy() )
                                                                    # color=vis_util.make_colors(
                                                                    #     torch.sigmoid(predictions[0][0])) ))
coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])
