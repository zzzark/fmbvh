Here is a simple guide to help you get started with using this package. 

```
from fmbvh.bvh.parser import BVH


# open a bvh file using the fmbvh parser, and print the information of it
filepath = "/path/to/your/bvh/file"
obj = BVH(filepath)
print(f"""
    fps: {obj.fps}
    all joint names: {obj.names}
    topology (represented as parent's indices): {obj.p_index},
    #frames: {obj.frames}
""")
print("------------")

# get some useful features
from fmbvh.motion_tensor.bvh_casting import get_offsets_from_bvh, get_quaternion_from_bvh

offsets = get_offsets_from_bvh(obj)
translation, quaternions = get_quaternion_from_bvh(obj)
print("------------")


# all the features are store in a JxDxT tensor, where J is the number of joints, D is the dimension, and T is the time length
print(offsets.shape)
print(translation.shape)
print(quaternions.shape)
print("  ^ NOTE: quaternions are in WXYZ order")
print("------------")


# to use 6d rotation as feature 
from fmbvh.motion_tensor.rotations import quaternion_to_matrix, matrix_to_rotation_6d
matrices = quaternion_to_matrix(quaternions)
rotation_6d = matrix_to_rotation_6d(matrices)
print(rotation_6d.shape)
print("------------")


# to use differential forward kinematics
from fmbvh.motion_tensor.kinematics import forward_kinematics

positions = forward_kinematics(obj.p_index, matrices, translation, offsets)
print(positions.shape)
print("------------")


# for visualization, simply run
from fmbvh.visualization.utils import show_bvh, bvh_to_video
print("""
    A S D W Z X: camera control
    Q: quit
""")
show_bvh(filepath, backend_cv=True)
bvh_to_video(filepath, "output.mp4")

```
