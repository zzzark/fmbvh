Here is a simple guide to help you get started with using this package. 

```
from fmbvh.bvh.parser import BVH

# Open a BVH file using the fmbvh parser and print its information
filepath = "/path/to/your/bvh/file"
obj = BVH(filepath)
print(f"""
    FPS: {obj.fps}
    Joint Names: {obj.names}
    Topology (parent indices): {obj.p_index}
    Number of Frames: {obj.frames}
""")
print("------------")

# Extract useful features
from fmbvh.motion_tensor.bvh_casting import get_offsets_from_bvh, get_quaternion_from_bvh

offsets = get_offsets_from_bvh(obj)
translation, quaternions = get_quaternion_from_bvh(obj)
print("------------")

# Features are stored in a JxDxT tensor, where J is the number of joints, D is the dimension, and T is the time length
print(offsets.shape)
print(translation.shape)
print(quaternions.shape)
print("  ^ NOTE: Quaternions are in WXYZ order")
print("------------")

# Use 6D rotation as a feature
from fmbvh.motion_tensor.rotations import quaternion_to_matrix, matrix_to_rotation_6d

matrices = quaternion_to_matrix(quaternions)
rotation_6d = matrix_to_rotation_6d(matrices)
print(rotation_6d.shape)
print("------------")

# Apply differential forward kinematics
from fmbvh.motion_tensor.kinematics import forward_kinematics

positions = forward_kinematics(obj.p_index, matrices, translation, offsets)
print(positions.shape)
print("------------")

# For visualization, simply run:
from fmbvh.visualization.utils import show_bvh, bvh_to_video

print("""
    Camera Controls: A S D W Z X
    Q: Quit
""")
show_bvh(filepath, backend_cv=True)
bvh_to_video(filepath, "output.mp4")


```
