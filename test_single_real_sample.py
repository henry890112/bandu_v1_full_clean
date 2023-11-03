import torch
from utils import misc_util, train_util, surface_util, vis_util
import open3d
import argparse
import numpy as np
import json
import open3d as o3d
from utils import color_util
from scipy.spatial.transform.rotation import Rotation as R
import copy


parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('sc_checkpoint', help="Model checkpoint: surface_classifier")
parser.add_argument('--sample_path')
parser.add_argument('--stats_json')
parser.add_argument('--device_id', default=0)

args = parser.parse_args()

#Henry: add
def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0.2,1.8], segmap_id=0, skip_border_objects=False, margin_px=5):
        """
        Converts depth map + intrinsics to point cloud. 
        If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

        Arguments:
            depth {np.ndarray} -- HxW depth map in m
            K {np.ndarray} -- 3x3 camera Matrix

        Keyword Arguments:
            segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
            rgb {np.ndarray} -- HxW rgb image (default: {None})
            z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
            segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
            skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
            margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

        Returns:
            [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
        """

        if K is None:
            raise ValueError('K is required either as argument --K or from the input numpy file')
            
        # Convert to pc 
        pc_full, pc_colors = depth2pc(depth, K, rgb)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
        pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
        
        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if segmap is not None:
            pc_segments = {}
            obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
            for i in obj_instances:
                if skip_border_objects and not i==segmap_id:
                    obj_i_y, obj_i_x = np.where(segmap==i)
                    if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                        print('object {} not entirely in image bounds, skipping'.format(i))
                        continue
                inst_mask = segmap==i
                pc_segment,_ = depth2pc(depth*inst_mask, K)
                pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

        return pc_full, pc_segments, pc_colors



config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)

models_dict = train_util.model_creator(config=config,
                            device_id=args.device_id)

sd = torch.load(args.sc_checkpoint, map_location="cpu")

models_dict['surface_classifier'].load_state_dict(sd['model'])

# models_dict['surface_classifier'].gpu_0 = torch.device(f"cuda:{args.gpu0}")
# models_dict['surface_classifier'].gpu_1 = torch.device(f"cuda:{args.gpu1}")


# load sample
# sample_path = "/root/bandu_v1_full_clean/out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples/Knight Shape/1.pkl"

# sample_path = "/root/bandu_v1_full_clean/out/aggregate_pc.torch"

# sample_pkl = torch.load(args.sample_path)

predictions_num_z = 1

batch = dict()

# num_points, 3 -> 1, 1, num_points, 3


# this is the real image pkl
#Henry 註解掉
# pcd = vis_util.make_point_cloud_o3d(sample_pkl['points'],
#                                     color=sample_pkl['colors'])

#Henry test pcd
from PIL import Image

sample_pkl = dict()

table_rgb1 = Image.open('./my_test_data/color_table1.png')
table_depth1 = np.load('./my_test_data/depth_table1.npy')
table_mask1 = Image.open('./my_test_data/mask_table1.png')
table_rgb2 = Image.open('./my_test_data/color_table2.png')
table_depth2 = np.load('./my_test_data/depth_table2.npy')
table_mask2 = Image.open('./my_test_data/mask_table2.png')
table_rgb3 = Image.open('./my_test_data/color_table3.png')
table_depth3 = np.load('./my_test_data/depth_table3.npy')
table_mask3 = Image.open('./my_test_data/mask_table3.png')

rgb_list = [table_rgb1, table_rgb2, table_rgb3]
depth_list = [table_depth1, table_depth2, table_depth3]
mask_list = [table_mask1, table_mask2, table_mask3]

#Henry can change this id to choose different object in the list
#Henry 0: mustard(pile)應該是底部太不均勻才會辨識失敗, 1: creacker_box(pile), 2: can(pack)成功topk=300up
chose_id = 2
#Henry用來選擇要多少點來決定平面的
# topk_k = 500

# table_rgb = np.array(rgb_list[chose_id])
# table_rgb = table_rgb/255
# table_depth = np.array(depth_list[chose_id])
# table_mask = np.array(mask_list[chose_id])


# intrinsic_matrix = np.array([[320, 0, 320],
#                              [0, 320, 320],
#                              [0, 0, 1]])


# table_pc_full, table_pc_segments, table_pc_colors = extract_point_clouds(depth = table_depth,   K = intrinsic_matrix, segmap = table_mask, rgb = table_rgb, z_range = [0.2, 1.])
# origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
# print(type(table_pc_segments[1]))
# print(table_pc_segments[1].shape)

# sample_pkl['points'] = table_pc_segments[1]

# # visulaize the point cloud
# table_pcd = o3d.geometry.PointCloud()
# table_pcd.points = o3d.utility.Vector3dVector(table_pc_full)
# table_pcd.colors = o3d.utility.Vector3dVector(table_pc_colors)
# # o3d.visualization.draw_geometries([table_pcd, origin_frame])

# table_seg_pcd = o3d.geometry.PointCloud()
# table_seg_pcd.points = o3d.utility.Vector3dVector(table_pc_segments[1])
# #change the color of the point cloud segments
# table_seg_pcd.paint_uniform_color([0, 0, 0])
# # o3d.visualization.draw_geometries([table_seg_pcd, origin_frame])
# pcd = table_seg_pcd
#Henry test pcd over

#Henry use my multiview pcd
# load the npy file
topk_k = 500

multiview_path = './my_test_data/pcd_combine_augmented.npy'
multiview_pcd = np.load(multiview_path)
sample_pkl['points'] = multiview_pcd

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(multiview_pcd)
pcd.paint_uniform_color([0, 0, 0])
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, origin_frame])
#Henry use my multiview pcd over

obb = open3d.geometry.OrientedBoundingBox()
obb = obb.create_from_points(pcd.points)

# center at COM

# if args.uniform_scale_longest_axis:
#     # calculate scale along longest axis
#     pcd.points = open3d.utility.Vector3dVector(np.array(sample_pkl['points']) - obb.get_center())
# else:

object_com = obb.get_center()

# center pointcloud
# pcd.points = open3d.utility.Vector3dVector(np.array(sample_pkl['points']) - object_com)

# Henry center
pcd.translate(-obb.get_center())

# visualize pointcloud
open3d.visualization.draw_geometries([pcd, origin_frame])


#Henry 註解掉
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.005)
# print the shape of pcd
# downsampled_pcd = pcd
print(np.array(downsampled_pcd.points).shape)
batch['rotated_pointcloud'] = torch.from_numpy(np.array(downsampled_pcd.points)).unsqueeze(0).unsqueeze(0)
# assert batch['rotated_pointcloud'].shape[2] > 1024 and batch['rotated_pointcloud'].shape[2] < 2048, batch['rotated_pointcloud'].shape


# below is if we have the training file pkl
# batch['rotated_pointcloud'] = torch.from_numpy(sample_pkl['rotated_pointcloud']).unsqueeze(0).unsqueeze(0)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)
batch['rotated_pointcloud_mean'] = torch.Tensor(stats_dic['rotated_pointcloud_mean'])

batch['rotated_pointcloud_var'] = torch.as_tensor(stats_dic['rotated_pointcloud_var'])

models_dict['surface_classifier'].eval()

predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                             z_samples_per_sample=predictions_num_z)

#Henry 介紹
# batch['rotated_pointcloud'][0][0] 為downsampled_pcd  (N, 3)
# predictions[0][0] 為預測的confidence (N, 1)
print(predictions[0][0])
print(predictions[0][0].shape)
'''
#visualize red and black pointcloud
open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                              color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1)) )])

# visualize confidence color mapped pointcloud
open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                              color=vis_util.make_color_map(torch.sigmoid(predictions[0][0]).squeeze(-1)) )])

# visualize thresholded pointcloud
open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                              color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1) ,
                                                background_color=color_util.MURKY_GREEN,
                                                surface_color=color_util.YELLOW))])
'''
#total visualize
open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                              color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                              vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0]+ torch.tensor([0, 0, .5]),
                              color=vis_util.make_color_map(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                              vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0]+ torch.tensor([0, 0, 1]),
                              color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1) ,
                                                background_color=color_util.MURKY_GREEN,
                                                surface_color=color_util.YELLOW))])


# TODO: was this trained on threshold 0 or .5?
#Henry 利用topk或sigmoid threshold來找出最好的平面  （二選一）
rotmat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][0][0],
                                                                            predictions[0][0],
                                                                            topk_k=topk_k
                                                                            # ,
                                                                            # sigmoid_threshold=0
                                                                            )

geoms_to_draw = []


box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .1])
norm_arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                             position=box_centroid,
                                           # object_com=sample_pkl['position'])
                                             object_com=np.zeros(3)) # because the object has been centered
geoms_to_draw.append(norm_arrow)
geoms_to_draw.append(box)


if 'points_incl_table' in sample_pkl.keys():
    scene_pcd = vis_util.make_point_cloud_o3d(sample_pkl['points_incl_table'],
                                        color=sample_pkl['colors_incl_table'],
                                              normalize_color=False)
    # visualize pointcloud
    large_coord_frame =  open3d.geometry.TriangleMesh.create_coordinate_frame(.1, [0, 0, 0])
    open3d.visualization.draw_geometries([scene_pcd,
                                         # large_coord_frame,
                                          copy.deepcopy(norm_arrow).translate(object_com),
                                          copy.deepcopy(box).translate(object_com)])


# below creates the cp binary mask from topk
# -> [num_points]
contact_points_binary_mask = torch.zeros_like(predictions[0][0].squeeze())
contact_points_binary_mask[torch.topk(predictions[0][0].squeeze(), topk_k, largest=False)[-1]] = 1

surface_points_binary_mask = 1 - contact_points_binary_mask
# below creates the contact points binary mask from sigmoid with threshold 50%
# surface_points_binary_mask = torch.round(torch.sigmoid(predictions[0][0]))
#
# contact_points_binary_mask = 1 -surface_points_binary_mask

# how to get colors
zeroed_out_colors = (torch.from_numpy(np.array(downsampled_pcd.colors)).to(surface_points_binary_mask.device) * surface_points_binary_mask.unsqueeze(-1))
original_rgb_with_red_contact_points = zeroed_out_colors + \
                                       contact_points_binary_mask.unsqueeze(-1) * torch.Tensor([1, 0, 0]).unsqueeze(0).expand(surface_points_binary_mask.shape[0], -1).to(contact_points_binary_mask.device)

# object_realsense_pcd = vis_util.make_point_cloud_o3d(sample_pkl['points'],
#                                     color=sample_pkl['colors'])
object_realsense_pcd = vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                                    color=original_rgb_with_red_contact_points.data.cpu().numpy() )
geoms_to_draw.append(object_realsense_pcd)
                                                                    # color=vis_util.make_colors(
                                                                    #     torch.sigmoid(predictions[0][0])) ))
coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])
geoms_to_draw.append(coord_frame)

# open3d.visualization.draw_geometries([geoms_to_draw, coord_frame, object_realsense_pcd])

open3d.visualization.draw([
    # {'name': 'object_mesh', 'geometry': object_mesh.transform(reg_p2p.transformation), 'material': mat},
                           {'name': 'coordinate_frame', 'geometry': coord_frame},
                           {'name': 'object_realsense_pcd', 'geometry': object_realsense_pcd},
                           {'name': 'norm_arrow', 'geometry': norm_arrow}],
                        #    {'name': 'box', 'geometry': box}],
                          show_skybox=False)
#Henry尋找平面結束
# icp visualization
mesh_path = "parts/stls/main/engmikedset/Nut.stl"

# resize to 0.75

object_mesh = open3d.io.read_triangle_mesh(mesh_path)

object_mesh.scale(.7, center=np.zeros(3))


object_mesh.paint_uniform_color(np.array([64,224,208])/255)
# object_mesh.paint_uniform_color(np.array([0, 0, 1]))

object_mesh.compute_vertex_normals()

object_mesh_pcd = object_mesh.sample_points_uniformly(number_of_points=1024)

geoms_to_draw.append(object_mesh)
mat = open3d.visualization.rendering.MaterialRecord()
mat.base_color = np.array([1, 1, 1, .8])
mat.shader = "defaultLitTransparency"


# open3d.visualization.draw_geometries(geoms_to_draw)

trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
threshold = 0.02

reg_p2p = o3d.pipelines.registration.registration_icp(
    object_mesh_pcd, object_realsense_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

# open3d.visualization.draw([
#     # {'name': 'object_mesh', 'geometry': object_mesh.transform(reg_p2p.transformation), 'material': mat},
#                            {'name': 'coordinate_frame', 'geometry': coord_frame},
#                            {'name': 'object_realsense_pcd', 'geometry': object_realsense_pcd},
#                            # {'name': 'norm_arrow', 'geometry': norm_arrow},
#                            {'name': 'box', 'geometry': box}],
#                           show_skybox=False)

# open3d.visualization.draw({'name': 'test', 'geometry': mesh, 'material': mat})


# draw the object after transformed, overlayed on the same table
table_rotation_x = .03
filter_height = -.13

workspace_limits = np.asarray([[0.6384-.25, 0.6384+.25], [.1325-.35, .1325+.35], [filter_height, filter_height+.2]])

# remove table points
table_normal = np.array([0, 0, 1])

table_height_vector = table_normal * filter_height

# rotate about x axis
new_normal = R.from_euler("x", table_rotation_x).apply(table_normal)

# filter out table points
mask = (np.array(scene_pcd.points) @ new_normal < workspace_limits[2][0])

# add back transformed object

# make correction so that object must be above table by offsetting difference between lowest point on pc and the table minimum
transformed_object_pts = (R.from_matrix(rotmat).apply(sample_pkl['points'] - object_com)  + object_com).copy()

lowest_transformed_object_pt = transformed_object_pts[transformed_object_pts[:, -1].argmin(), :]

if lowest_transformed_object_pt[-1] < workspace_limits[2][0]:
    offset = workspace_limits[2][0] - lowest_transformed_object_pt[-1]
    transformed_object_pts += np.array([0, 0, offset])

new_scene_pc = np.concatenate([np.array(scene_pcd.points)[mask, :],
                               transformed_object_pts])

new_scene_colors = np.concatenate([np.array(scene_pcd.colors)[mask, :],
                                   sample_pkl['colors']])

scene_pcd.points = open3d.utility.Vector3dVector(new_scene_pc)
scene_pcd.colors = open3d.utility.Vector3dVector(new_scene_colors)



open3d.visualization.draw_geometries([scene_pcd])
                                      # large_coord_frame])