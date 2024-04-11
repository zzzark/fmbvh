# O3D (LEGACY)
from .o3drnd import quick_visualize, quick_visualize_fk

# OPENCV
from .cvrnd import render_pose


def show_bvh(bvh_file, scale=200, backend_cv=False):
    from ..bvh.parser import BVH

    if not isinstance(bvh_file, BVH):
        bvh_file = BVH(bvh_file)
    
    from ..motion_tensor.bvh_casting import get_positions_from_bvh
    pos = get_positions_from_bvh(bvh_file)

    if backend_cv:
        render_pose(bvh_file.p_index, pos, None, bvh_file.fps)
    else:
        quick_visualize(bvh_file.p_index, pos, scale)
