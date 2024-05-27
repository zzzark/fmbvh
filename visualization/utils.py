# O3D (LEGACY)
fn_dummy = lambda *_, **__: None

try:
    from .o3drnd import quick_visualize, quick_visualize_fk
except ImportError as e:
    print(e, ", omitting it")
    quick_visualize, quick_visualize_fk = fn_dummy, fn_dummy

# OPENCV
try:
    from .cvrnd import render_pose
except ImportError as e:
    print(e, ", omitting it")
    render_pose = fn_dummy


def show_bvh(bvh_file, scale=None, backend_cv=False, **kwargs):
    from ..bvh.parser import BVH

    if not isinstance(bvh_file, BVH):
        bvh_file = BVH(bvh_file)
    
    from ..motion_tensor.bvh_casting import get_positions_from_bvh
    pos = get_positions_from_bvh(bvh_file)

    if backend_cv:
        render_pose(bvh_file.p_index, pos, None, bvh_file.fps, scale=scale, **kwargs)
    else:
        if scale is None: scale=200
        quick_visualize(bvh_file.p_index, pos, scale)


def bvh_to_video(bvh_file, output_file, format="avc1"):
    from ..bvh.parser import BVH

    if not isinstance(bvh_file, BVH):
        bvh_file = BVH(bvh_file)

    from ..motion_tensor.bvh_casting import get_positions_from_bvh
    pos = get_positions_from_bvh(bvh_file)
    render_pose(bvh_file.p_index, pos, output=output_file, fps=bvh_file.fps, format=format)
