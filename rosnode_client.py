#!/usr/bin/env python3

import rospy
import numpy as np
from bandu_v1_full_clean.srv import GetTargetMatrix, GetTargetMatrixRequest
from std_msgs.msg import Float32MultiArray
import os
from std_srvs.srv import Empty

def shutdown_server():
    # rospy.init_node('shutdown_client')
    rospy.wait_for_service('shutdown')
    try:
        shutdown_service = rospy.ServiceProxy('shutdown', Empty)
        shutdown_service()
        rospy.loginfo("Shutdown request sent.")
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s" % e)

def target_matrix_client():
    rospy.init_node('target_matrix_client_node')

    # 等待服務可用
    rospy.wait_for_service('get_target_matrix')
    
    try:
        get_target_matrix = rospy.ServiceProxy('get_target_matrix', GetTargetMatrix)
        
        # 構建請求
        req = GetTargetMatrixRequest()
        
        # 假設multiview_pc_target_base是一個包含點雲數據的列表，這裡用隨機數據作為示例
        parent_directory = os.path.join(os.path.dirname(__file__))
        result_list = ['blue_bottle', 'mug', 'small_mug', 'long_box', 'tap', 'wash_hand']
        multiview_path = os.path.join(parent_directory, f'my_test_data/{result_list[0]}.npy')
        multiview_pointcloud = np.load(multiview_path)
        multiview_pc_target_base = multiview_pointcloud.flatten().tolist()
        req.multiview_pc_target_base = multiview_pc_target_base
        
        # 呼叫服務並獲取響應
        resp = get_target_matrix(req)
        
        # 打印接收到的目標矩陣
        print("Received target matrix:")
        print(np.array(resp.target_matrix).reshape(4, 4))
    
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    target_matrix_client()
    # shutdown_server()
