import pybullet as p
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('urdf_path',
                    type=str,
                    help='URDF Path')

args = parser.parse_args()



p.connect(p.GUI)
obj_id = p.loadURDF(args.urdf_path, globalScaling=1)

# Bandu Block
# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi/2, np.pi/2]), 0.0, np.random.uniform(0, 2*np.pi)]).as_quat())

# Skewed Rectangular Prism
# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi, 0, np.pi]), np.random.choice([-np.pi, np.pi, 0]), np.random.uniform(0, 2*np.pi)]).as_quat())

# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi]), np.random.choice([-np.pi]), np.random.uniform(0, 2*np.pi)]).as_quat())

aabb = p.getAABB(obj_id)

# draw the aabb box in the pybullet GUI

print("aabb")
print(aabb)

aabbMinVec = aabb[0]
aabbMaxVec = aabb[1]
print("aabbMinVec")
print(aabbMinVec)
print("aabbMaxVec")
print(aabbMaxVec)
# Set the color and line width for the AABB box
color = [1, 0, 0]  # Red color
line_width = 2.0

# draw the AABB box
# 底部

p.addUserDebugLine(aabbMinVec, [aabbMinVec[0], aabbMinVec[1], aabbMaxVec[2]], color, line_width)
p.addUserDebugLine(aabbMinVec, [aabbMinVec[0], aabbMaxVec[1], aabbMinVec[2]], color, line_width)
p.addUserDebugLine(aabbMinVec, [aabbMaxVec[0], aabbMinVec[1], aabbMinVec[2]], color, line_width)
p.addUserDebugLine(aabbMaxVec, [aabbMaxVec[0], aabbMaxVec[1], aabbMinVec[2]], color, line_width)
p.addUserDebugLine(aabbMaxVec, [aabbMaxVec[0], aabbMinVec[1], aabbMaxVec[2]], color, line_width)
p.addUserDebugLine(aabbMaxVec, [aabbMinVec[0], aabbMaxVec[1], aabbMaxVec[2]], color, line_width)

p.addUserDebugLine([aabbMaxVec[0], aabbMinVec[1], aabbMaxVec[2]], [aabbMinVec[0], aabbMinVec[1], aabbMaxVec[2]], color, line_width)
p.addUserDebugLine([aabbMaxVec[0], aabbMinVec[1], aabbMaxVec[2]], [aabbMaxVec[0], aabbMinVec[1], aabbMinVec[2]], color, line_width)
p.addUserDebugLine([aabbMaxVec[0], aabbMaxVec[1], aabbMinVec[2]], [aabbMaxVec[0], aabbMinVec[1], aabbMinVec[2]], color, line_width)

p.addUserDebugLine([aabbMinVec[0], aabbMaxVec[1], aabbMaxVec[2]], [aabbMinVec[0], aabbMinVec[1], aabbMaxVec[2]], color, line_width)
p.addUserDebugLine([aabbMinVec[0], aabbMaxVec[1], aabbMaxVec[2]], [aabbMinVec[0], aabbMaxVec[1], aabbMinVec[2]], color, line_width)
p.addUserDebugLine([aabbMaxVec[0], aabbMaxVec[1], aabbMinVec[2]], [aabbMinVec[0], aabbMaxVec[1], aabbMinVec[2]], color, line_width)



print("obj height")
print(aabbMaxVec[-1] - aabbMinVec[-1])
while 1:
    pass