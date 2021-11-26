import pybullet as p
import pickle
import numpy as np

GREEN = [0, 1, 0, 1.0]
GRAY = [.8, .8, .8, 1.0]
BLUE = [0, 0, 1, 1.0]
RED = [1, 0,0,1]

BLUE_TRANSPARENT = [0, 0, 1, .5]
GREEN_TRANSPARENT = [0, 1, 0, .5]
GRAY_TRANSPARENT = [.8, .8, .8, .5]
BLACK_TRANSPARENT = [0, 0, 0, .5]


def get_cam_img(cam_pkl, scale_factor=1/2):
    with open(f"{cam_pkl}", "rb") as fp:
        camStateList = pickle.load(fp)
        width, \
        height, \
        viewMatrix, \
        projectionMatrix, \
        cameraUp, \
        cameraForward, \
        horizontal, \
        vertical, \
        yaw, \
        pitch, \
        dist, \
        target = camStateList
        img_width, img_height, rgbPixels, depthPixels, segmentationMaskBufffer = p.getCameraImage(width=int(width*scale_factor),
                                                                                                  height=int(height*scale_factor),
                                                                                                  viewMatrix=viewMatrix,
                                                                                                  projectionMatrix=projectionMatrix,
                                                                                                  flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

        depth_buffer = np.reshape(depthPixels, [img_height,
                                                img_width])
        # znear = self.cfgs.CAM.SIM.ZNEAR
        # zfar = self.cfgs.CAM.SIM.ZFAR
        # depth = zfar * znear / (zfar - (zfar - znear) * depth_buffer)
        # seg = np.reshape(images[4], [self.img_height,
        #                              self.img_width])
        return [img_width, img_height, rgbPixels, depthPixels, segmentationMaskBufffer]


def load_geom(shape_type, size=None, mass=0.5, visualfile=None,
              collifile=None, mesh_scale=None, rgba=None,
              specular=None, shift_pos=None, shift_ori=None,
              base_pos=None, base_ori=None, **kwargs):
    """
    Load a regular geometry (`sphere`, `box`,
    `capsule`, `cylinder`, `mesh`).

    Note:
        Please do not call **load_geom('capsule')** when you are using
        **robotiq gripper**. The capsule generated will be in wrong size
        if the mimicing thread (_th_mimic_gripper) in the robotiq
        gripper class starts running.
        This might be a PyBullet Bug (current version is 2.5.6).
        Note that other geometries(box, sphere, cylinder, etc.)
        are not affected by the threading in the robotiq gripper.

    Args:
        shape_type (str): one of [`sphere`, `box`, `capsule`, `cylinder`,
            `mesh`].

        size (float or list): Defaults to None.

             If shape_type is sphere: size should be a float (radius).

             If shape_type is capsule or cylinder: size should be
             a 2-element list (radius, length).

             If shape_type is box: size can be a float (same half
             edge length for 3 dims) or a 3-element list
             containing the half size of 3 edges

             size doesn't take effect if shape_type is mesh.

        mass (float): mass of the object in kg.
            If mass=0, then the object is static.

        visualfile (str): path to the visual mesh file.
            only needed when the shape_type is mesh. If it's None, same
            collision mesh file will be used as the visual mesh file.

        collifile (str): path to the collision mesh file.
            only needed when the shape_type is mesh. If it's None, same
            viusal mesh file will be used as the collision mesh file.

        mesh_scale (float or list): scale the mesh. If it's a float number,
            the mesh will be scaled in same ratio along 3 dimensions.
            If it's a list, then it should contain 3 elements
            (scales along 3 dimensions).
        rgba (list): color components for red, green, blue and alpha,
            each in range [0, 1] (shape: :math:`[4,]`).
        specular(list): specular reflection color components
            for red, green, blue and alpha, each in
            range [0, 1] (shape: :math:`[4,]`).
        shift_pos (list): translational offset of collision
            shape, visual shape, and inertial frame (shape: :math:`[3,]`).
        shift_ori (list): rotational offset (quaternion [x, y, z, w])
            of collision shape, visual shape, and inertial
            frame (shape: :math:`[4,]`).
        base_pos (list): cartesian world position of
            the base (shape: :math:`[3,]`).
        base_ori (list): cartesian world orientation of the base as
            quaternion [x, y, z, w] (shape: :math:`[4,]`).

    Returns:
        int: a body unique id, a non-negative integer
        value or -1 for failure.

    """
    global GRAVITY_CONST
    pb_shape_types = {'sphere': p.GEOM_SPHERE,
                      'box': p.GEOM_BOX,
                      'capsule': p.GEOM_CAPSULE,
                      'cylinder': p.GEOM_CYLINDER,
                      'mesh': p.GEOM_MESH}
    if shape_type not in pb_shape_types.keys():
        raise TypeError('The following shape output_type is not '
                        'supported: %s' % shape_type)

    collision_args = {'shapeType': pb_shape_types[shape_type]}
    visual_args = {'shapeType': pb_shape_types[shape_type]}
    if shape_type == 'sphere':
        if size is not None and not (isinstance(size, float) and size > 0):
            raise TypeError('size should be a positive '
                            'float number for a sphere.')
        collision_args['radius'] = 0.5 if size is None else size
        visual_args['radius'] = collision_args['radius']
    elif shape_type == 'box':
        if isinstance(size, float):
            size = [size, size, size]
        elif isinstance(size, list):
            if len(size) != 3:
                raise ValueError('If size is a list, its length'
                                 ' should be 3 for a box')
        elif size is not None:
            raise TypeError('size should be a float number, '
                            'or a 3-element list '
                            'for a box')
        collision_args['halfExtents'] = [1, 1, 1] if size is None else size
        visual_args['halfExtents'] = collision_args['halfExtents']
    elif shape_type in ['capsule', 'cylinder']:
        if size is not None:
            if not isinstance(size, list) or len(size) != 2:
                raise TypeError('size should be a 2-element '
                                'list (radius, length)'
                                'for a capsule or a cylinder.')
            for si in size:
                if not isinstance(si, Number) or si <= 0.0:
                    raise TypeError('size should be a list that '
                                    'contains 2 positive'
                                    'numbers (radius, length) for '
                                    'a capsule or '
                                    'a cylinder.')
        collision_args['radius'] = 0.5 if size is None else size[0]
        visual_args['radius'] = collision_args['radius']
        collision_args['height'] = 1.0 if size is None else size[1]
        visual_args['length'] = collision_args['height']
    elif shape_type == 'mesh':
        if visualfile is None and collifile is None:
            raise ValueError('At least one of the visualfile and collifile'
                             'should be provided!')
        if visualfile is None:
            visualfile = collifile
        elif collifile is None:
            collifile = visualfile
        if not isinstance(visualfile, str):
            raise TypeError('visualfile should be the path to '
                            'the visual mesh file!')
        if not isinstance(collifile, str):
            raise TypeError('collifile should be the path to '
                            'the collision mesh file!')
        collision_args['fileName'] = collifile
        visual_args['fileName'] = visualfile
        if isinstance(mesh_scale, float):
            mesh_scale = [mesh_scale, mesh_scale, mesh_scale]
        elif isinstance(mesh_scale, list):
            if len(mesh_scale) != 3:
                raise ValueError('If mesh_scale is a list, its length'
                                 ' should be 3.')
        elif mesh_scale is not None:
            raise TypeError('mesh_scale should be a float number'
                            ', or a 3-element list.')
        collision_args['meshScale'] = [1, 1, 1] if mesh_scale is None \
            else mesh_scale
        visual_args['meshScale'] = collision_args['meshScale']
    else:
        raise TypeError('The following shape output_type is not '
                        'supported: %s' % shape_type)

    visual_args['rgbaColor'] = rgba
    visual_args['specularColor'] = specular
    collision_args['collisionFramePosition'] = shift_pos
    collision_args['collisionFrameOrientation'] = shift_ori
    visual_args['visualFramePosition'] = shift_pos
    visual_args['visualFrameOrientation'] = shift_ori

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    vs_id = p.createVisualShape(**visual_args)
    cs_id = p.createCollisionShape(**collision_args)
    body_id = p.createMultiBody(baseMass=mass,
                                   baseInertialFramePosition=shift_pos,
                                   baseInertialFrameOrientation=shift_ori,
                                   baseCollisionShapeIndex=cs_id,
                                   baseVisualShapeIndex=vs_id,
                                   basePosition=base_pos,
                                   baseOrientation=base_ori,
                                   **kwargs)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # p.setGravity(0, 0, GRAVITY_CONST)
    return body_id


