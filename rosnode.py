#!/usr/bin/env python3
import sys
import rospy
import torch
from utils import misc_util, train_util, surface_util, vis_util
from utils.vis_util import make_color_map, make_colors, make_colors_topk
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from std_msgs.msg import Float32MultiArray
from bandu_v1_full_clean.srv import GetTargetMatrix, GetTargetMatrixResponse  # 根據你的package名稱調整
from utils import misc_util, train_util, surface_util, vis_util
from std_srvs.srv import Empty ,EmptyResponse  



def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
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


class TargetMatrixService:
    def __init__(self):
        self.device_id = 0
        self.vis = True
        self.vis_pcd = True

        self.topk_k = 100
        self.get_toward_vector = True
        self.fps_pc = False

        self.hyper_config = rospy.get_param('~hyper_config')
        self.sc_checkpoint = rospy.get_param('~sc_checkpoint')
        self.stats_json = rospy.get_param('~stats_json')

        self.config = misc_util.load_hyperconfig_from_filepath(self.hyper_config)
        self.models_dict = train_util.model_creator(config=self.config, device_id=self.device_id)

        sd = torch.load(self.sc_checkpoint, map_location="cpu")
        self.models_dict['surface_classifier'].load_state_dict(sd['model'])
        self.models_dict['surface_classifier'].eval()
        
        with open(self.stats_json, "r") as fp:
            self.stats_dic = json.load(fp)

        self.target_matrix_pub = rospy.Publisher('/target_matrix', Float32MultiArray, queue_size=10)
        self.service = rospy.Service('get_target_matrix', GetTargetMatrix, self.handle_get_target_matrix)
        self.shutdown_service = rospy.Service('shutdown', Empty, self.handle_shutdown)
        rospy.loginfo("Init finished")
        

    def handle_shutdown(self, req):
        rospy.loginfo("Shutdown service called, shutting down...")
        rospy.signal_shutdown("Shutdown requested")
        return EmptyResponse()
    
    def handle_get_target_matrix(self, req):
        multiview_pointcloud = np.array(req.multiview_pc_target_base).reshape(-1, 3)
        origin_center = np.mean(multiview_pointcloud, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(multiview_pointcloud)
        pcd.paint_uniform_color([0, 0, 0])
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        origin_center = np.mean(multiview_pointcloud, axis=0)
        pcd.translate(-origin_center)
        points = np.asarray(pcd.points)

        point_state = regularize_pc_point_count(points, 2048, use_farthest_point=self.fps_pc)
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(point_state)
        downsampled_pcd.paint_uniform_color([0, 0, 0])

        batch = dict()
        batch['rotated_pointcloud_mean'] = torch.Tensor(self.stats_dic['rotated_pointcloud_mean'])
        batch['rotated_pointcloud_var'] = torch.as_tensor(self.stats_dic['rotated_pointcloud_var'])
        batch['rotated_pointcloud'] = torch.from_numpy(np.array(downsampled_pcd.points)).unsqueeze(0).unsqueeze(0)
        predictions = self.models_dict['surface_classifier'].decode_batch(batch, ret_eps=False, z_samples_per_sample=1)
    
        print('=====predictions=====:', predictions[0][0].shape)

        rotmat, plane_model, predicted_cluster_pcd = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][0][0],
                                                                                    predictions[0][0],
                                                                                    topk_k=self.topk_k,
                                                                                    use_cluster=True)

        box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .1])
        arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                                position=box_centroid,
                                                object_com=np.zeros(3)) # because the object has been centered
        if self.vis_pcd:
            o3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                    color=vis_util.make_colors(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                                    vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0]+ torch.tensor([0, 0, .5]),
                                    color=vis_util.make_color_map(torch.sigmoid(predictions[0][0]).squeeze(-1))),

                                    predicted_cluster_pcd,
                                    box,
                                    arrow])
        
        target_transform = np.eye(4)
        target_transform[:3, 2] = plane_model[:3]
        target_transform[:3, 1] = np.cross(plane_model[:3], np.random.rand(3))
        target_transform[:3, 0] = np.cross(plane_model[:3], target_transform[:3, 1])
        target_transform[:3, 3] = origin_center

        if self.get_toward_vector:
            center = torch.mean(batch['rotated_pointcloud'][0][0], axis=0)

            n = plane_model[:3]
            toward_point = batch['rotated_pointcloud'][0][0]
            norm = np.linalg.norm(n)

            projections = toward_point - np.outer(np.dot(toward_point, n) / norm**2, n)

            centroid = torch.mean(projections, axis=0)
            points_centered = projections - centroid
            cov_matrix = np.cov(points_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]

            second_principal_axis = eigenvectors[:, 1]

            target_transform[:3, 1] = second_principal_axis
            target_transform[:3, 0] = np.cross(plane_model[:3], second_principal_axis)

        z_arrow = vis_util.create_arrow_final(plane_model[:3], [0., 0., 1.], scale=.15,
                                                position=[0, 0, 0],
                                                object_com=np.zeros(3)) # because the object has been centered
        x_arrow = vis_util.create_arrow_final(second_principal_axis, [1., 0., 0.], scale=.15,
                                                    position=[0, 0, 0],
                                                    object_com=np.zeros(3)) # because the object has been centered
        y_arrow = vis_util.create_arrow_final(np.cross(plane_model[:3], second_principal_axis), [0., 1., 0.], scale=.15,
                                                    position=[0, 0, 0],
                                                    object_com=np.zeros(3)) # because the object has been centered


        origin_frame.translate(-origin_center)
        if self.vis_pcd: 
            o3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                                                            color=make_colors(torch.sigmoid(predictions[0][0]))),
                                                x_arrow,
                                                z_arrow,
                                                y_arrow,
                                                origin_frame])
        # 發布結果
        target_matrix_msg = Float32MultiArray(data=target_transform.flatten().tolist())
        self.target_matrix_pub.publish(target_matrix_msg)

        return GetTargetMatrixResponse(target_matrix=target_transform.flatten().tolist())


if __name__ == "__main__":
    rospy.init_node('target_matrix_service_node')
    service = TargetMatrixService()
    rospy.spin()
