from .parser import BVH, JointOffset, JointMotion
from collections import OrderedDict  # NOTE: since python >= 3.6, OrderedDict == dict
import itertools
from copy import deepcopy
from typing import Tuple, List, Dict
import torch

# TODO: optimize code structure
from ..motion_tensor import rotations as mor
from ..motion_tensor import bvh_casting as bvc


def reorder_bvh(obj: BVH):
    """
    reorder the offset_data and motion_data inplace (i.e. it will modify the `obj` object),
    since both of them use `OrderedDict` to store data and thus
    their keys need to be reordered after inserting or deleting
    some joints
    :param obj:
    :return:
    """
    dfs = [name for name, _ in obj.dfs()]
    for name in dfs:
        obj.offset_data.move_to_end(name)
        obj.motion_data.move_to_end(name)
    return obj


def build_bvh_from_scratch(p_index, bone_names, fps) -> BVH:
    """
    :param p_index: parent indices that determines the topology
    :param bone_names: bone names
    :param fps: frames per second, fps == 1.0 / frame_time
    """
    obj = BVH()
    obj.offset_data = OrderedDict()
    obj.motion_data = OrderedDict()
    obj.frames = 0
    obj.frame_time = 1.0 / fps
    obj.root_name = bone_names[0]
    obj.filepath = ""
    for i, name in enumerate(bone_names):
        if p_index[i] < 0:
            p_name = ""
            order = "XYZZYX"
            channel = 6
        else:
            p_name = bone_names[p_index[i]]
            order = "ZYX"
            channel = 3

        c_names = [bone_names[e] for e in range(len(p_index)) if p_index[e] == i]
        obj.offset_data[name] = JointOffset(name, p_name, c_names, [1, 1, 1], channel, order)  # set default bone offset to [1, 1, 1]
        obj.motion_data[name] = JointMotion(name, [])
    obj = reorder_bvh(obj)
    return obj


def reorder_joints(obj: BVH, parent_name: str, children_names_reordered: list):
    pj = obj.offset_data[parent_name]
    if len([None for name in pj.children_names if name not in children_names_reordered]):
        raise ValueError(f"should contain all the children joints: {pj.children_names}")

    pj.children_names = deepcopy(children_names_reordered)

    return reorder_bvh(obj)


def get_remaining_joint_names(obj: BVH, joint_names: list):
    names = [name for name, _ in obj.dfs()]
    return [name for name in names if name not in joint_names]


def rectify_joint(obj: BVH, parent: str, target: str, direction: list):
    """
    to correct a joint, e.g. A-pose to T-pose
    :param obj:
    :param parent:
    :param target:
    :param direction: e.g. direction [0, -1, 0] for `UpperLeg` joint forces it to face to the ground (A-pose to T-pose)
    :return:
    """
    QUA = mor.pad_position_to_quaternion
    MUL = mor.mul_two_quaternions
    INV = mor.inverse_quaternion
    V2Q = mor.quaternion_from_two_vectors
    POS = lambda x: x[:, 1:, :]
    NORM = mor.normalize_quaternion
    RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
    ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

    q_ls = [parent]

    def __gather_children(l_, n_):
        j_ = obj.offset_data[n_]
        l_ += j_.children_names
        for c_ in j_.children_names:
            __gather_children(l_, c_)

    __gather_children(q_ls, q_ls[0])

    pis = obj.get_indices_of_joints(q_ls)
    tis = pis[1:]

    t, q = bvc.get_quaternion_from_bvh(obj)
    o = bvc.get_offsets_from_bvh(obj)

    src = o[[pis[1]]]
    dst = torch.empty_like(src, dtype=src.dtype, device=src.device)
    dst[:, 0, :], dst[:, 1, :], dst[:, 2, :] = direction[0], direction[1], direction[2]
    Q = V2Q(src, dst)
    Q_ = INV(Q)

    Qx = q[pis]
    Qx = MUL(Qx, Q_)
    Qx[1:] = MUL(Q, Qx[1:])
    Qx = RECT(NORM(Qx))
    q[pis] = Qx[:]
    obj = bvc.write_quaternion_to_bvh(t, q, obj)

    Lx = QUA(o[tis])
    Lx = MUL(Q, MUL(Lx, Q_))
    o[tis] = POS(Lx)
    obj = bvc.write_offsets_to_bvh(o, obj)

    return obj


def cut_off_joints(obj: BVH, remove_names: list):

    def try_remove(name_):
        off: JointOffset = obj.offset_data[name_]
        if name_ == obj.root_name or len(off.children_names) != 0:
            raise ValueError(f"'{name_}' is not a leaf node.")
        par_ = off.parent_name
        p_off: JointOffset = obj.offset_data[par_]
        p_off.children_names.remove(name_)
        del obj.offset_data[name_]
        del obj.motion_data[name_]

    reversed_names = [name for name, _ in obj.dfs()][::-1]
    for fit_name, _ in filter(lambda x: x[0] == x[1], itertools.product(reversed_names, remove_names)):
        try_remove(fit_name)

    return reorder_bvh(obj)


def shift_joint(obj: BVH, target_name: str, offset: list):
    """
    change target joint offset to a new one
    :param obj:
    :param target_name:
    :param offset:
    :return:
    """
    QUA = mor.pad_position_to_quaternion
    MUL = mor.mul_two_quaternions
    INV = mor.inverse_quaternion
    V2Q = mor.quaternion_from_two_vectors
    POS = lambda x: x[:, 1:, :]
    NORM = mor.normalize_quaternion
    RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
    ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

    tj = obj.offset_data[target_name]
    pn = tj.parent_name
    tn = target_name
    cn = tj.children_names

    pi = obj.get_indices_of_joints([pn])  # parent index
    ti = obj.get_indices_of_joints([tn])  # target index
    ci = obj.get_indices_of_joints(cn)  # indices of children

    if len(ci) != 0:
        t, q = bvc.get_quaternion_from_bvh(obj)
        o = bvc.get_offsets_from_bvh(obj)

        Qp_, Qt_, Qc_ = q[pi], q[ti], q[ci]
        Lp_, Lt_, Lc_ = o[pi], o[ti], o[ci]
        Lp_, Lt_, Lc_ = QUA(Lp_), QUA(Lt_), QUA(Lc_)

        Ltp = torch.empty_like(Lt_, dtype=Lt_.dtype, device=Lt_.device)
        Ltp[:, 0, :], Ltp[:, 1, :], Ltp[:, 2, :], Ltp[:, 3, :] = 0.0, offset[0], offset[1], offset[2]
        Lcp = Lt_ + Lc_ - Ltp

        vec = ROTATE(Qp_, Lt_) + ROTATE(MUL(Qp_, Qt_), Lc_) - ROTATE(Qp_, Ltp)
        vec = MUL(INV(Qp_), MUL(vec, Qp_))
        tqp = V2Q(POS(Lcp), POS(vec))  # what the target rotation should be for every child
        tqp = tqp.mean(dim=0, keepdim=True)  # get the average result if there are multiple children
        tqp = RECT(NORM(tqp))
        q[ti] = tqp
        obj = bvc.write_quaternion_to_bvh(t, q, obj)
    else:
        pass  # just move the end-effector so there is no need to change

    # set the new target offset
    org = tj.offset
    tj.offset = [e for e in offset]
    for c in cn:
        cj = obj.offset_data[c]
        cj.offset = [a + b - c for a, b, c in zip(org, cj.offset, tj.offset)]
    return reorder_bvh(obj)


def remove_joint(obj: BVH, remove_names: Tuple[list, str], inherent='mul'):
    """
    remove a list of given joints,
    this is useful for removing joints that are at the same position
    :param obj:
    :param remove_names:
    :param inherent: inherent joint rotation: none, mul, recompute
    :return: edited bvh object
    """
    if isinstance(remove_names, str):
        remove_names = [remove_names]
    for rm_name in remove_names:
        if rm_name == obj.root_name:
            raise ValueError("cannot remove root joint!")

        tj = obj.offset_data[rm_name]  # target joint
        pn = tj.parent_name            # parent name
        tn = rm_name

        if inherent == 'none':
            ic = obj.get_indices_of_joints(obj.offset_data[rm_name].children_names)  # indices of children
            if len(ic) != 0:
                # add target offset to children
                for cn in tj.children_names:
                    cj = obj.offset_data[cn]
                    cj.offset = [a + b for a, b in zip(cj.offset, tj.offset)]
            else:
                pass
        elif inherent == 'mul':
            # add target motion to children
            MUL = mor.mul_two_quaternions
            NORM = mor.normalize_quaternion
            RECT = lambda x: mor.rectify_w_of_quaternion(x, True)

            t, q = bvc.get_quaternion_from_bvh(obj)

            it = obj.get_indices_of_joints([tn])  # target index
            ic = obj.get_indices_of_joints(obj.offset_data[rm_name].children_names)  # indices of children

            if len(ic) != 0:
                Qt, Qc = q[it], q[ic]
                q[ic] = RECT(NORM(MUL(Qt, Qc)))  # directly combine the two rotations
                obj = bvc.write_quaternion_to_bvh(t, q, obj)

                # add target offset to children
                for cn in tj.children_names:
                    cj = obj.offset_data[cn]
                    cj.offset = [a+b for a, b in zip(cj.offset, tj.offset)]
            else:
                pass
        elif inherent == 'recompute':
            # add target motion to children
            QUA = mor.pad_position_to_quaternion
            MUL = mor.mul_two_quaternions
            INV = mor.inverse_quaternion
            V2Q = mor.quaternion_from_two_vectors
            POS = lambda x: x[:, 1:, :]
            NORM = mor.normalize_quaternion
            RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
            ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

            t, q = bvc.get_quaternion_from_bvh(obj)
            o = bvc.get_offsets_from_bvh(obj)

            cns = obj.offset_data[rm_name].children_names  # children of target joint
            ip = obj.get_indices_of_joints([pn])  # parent index
            it = obj.get_indices_of_joints([tn])  # target index
            ics = obj.get_indices_of_joints(cns)  # indices of children

            if len(ics) != 0:
                for ic, cn in zip(ics, cns):
                    gns = obj.offset_data[cn].children_names  # grandchildren names
                    if len(gns) == 0:
                        continue

                    Qp, Qt, Qc = q[ip], q[it], q[ic]
                    Lp, Lt, Lc = o[ip], o[it], o[ic]
                    Lp, Lt, Lc = QUA(Lp), QUA(Lt), QUA(Lc)

                    igs = obj.get_indices_of_joints(gns)
                    Lg = o[igs].mean(dim=0, keepdims=True)  # get the average results as the succeeded joint's offset
                    Lg = QUA(Lg)

                    a = ROTATE(Qp, Lt) + ROTATE(MUL(Qp, Qt), Lc) + ROTATE(MUL(Qp, MUL(Qt, Qc)), Lg)
                    b = ROTATE(Qp, Lt + Lc)
                    vec = MUL(INV(Qp), MUL(a - b, Qp))
                    Qc_ = V2Q(POS(Lg), POS(vec))

                    q[ic] = RECT(NORM(Qc_))

                obj = bvc.write_quaternion_to_bvh(t, q, obj)
            else:
                pass  # use parent rotation

            # add target offset to children
            for cn in tj.children_names:
                cj = obj.offset_data[cn]
                cj.offset = [a+b for a, b in zip(cj.offset, tj.offset)]
        else:
            raise ValueError(f'inherent_rotation should be `none`, `mul` or `recompute`, not {inherent}')

        # ---- for parent ---- #
        pj = obj.offset_data[pn]           # parent joint
        pj.children_names.remove(rm_name)  # delete target joint
        pj.children_names += tj.children_names  # take over all the children of target joint

        # ---- for children ---- #
        # assign a new parent (i.e. target joint's parent)
        for cn in tj.children_names:  # cn: child name
            obj.offset_data[cn].parent_name = pn

        # ---- for target ---- #
        # delete target joint
        del obj.offset_data[rm_name]
        del obj.motion_data[rm_name]

        obj = reorder_bvh(obj)

    return obj


def insert_joint_between(obj: BVH, j1: str, j2: str, new_name: str, new_offset: Tuple[None, list]=None, divide_ratio=0.5):
    """
    :param obj: bvh object
    :param j1: name of joint 1
    :param j2: name of joint 2
    :param new_name: name of inserted joint
    :param new_offset: [Optional] new offset of the new joint
    :param divide_ratio: split between two (j1, j2), this parameter is ignored if new_offset is not None
    :return: edited bvh object (inplace!)
    """
    if new_name in obj.offset_data:
        raise ValueError(f"Name {new_name} already exists!")
    ja: JointOffset = obj.offset_data[j1]
    jb: JointOffset = obj.offset_data[j2]

    if j1 in jb.children_names:  # j1, ja: parent, j2, jb: child
        j1, j2 = j2, j1
        ja, jb = jb, ja
    elif j2 not in ja.children_names:
        raise ValueError(f"{j1} and {j2} should be father and child.")

    # input:   ja +---> jb ----> [...]
    #             |
    #             +---> jc
    #             |
    #             +---> [...]
    #
    # output:  ja +---> jn ----> jb ----> [...]
    #             |
    #             +---> jc
    #             |
    #             +---> [...]
    #
    # ---- offset data ---- #
    off_jn = [b*divide_ratio for b in jb.offset] if new_offset is None else new_offset
    off_jb = [b - a for a, b in zip(off_jn, jb.offset)]

    jn = deepcopy(jb)  # joint new
    jn.name = new_name
    jn.offset = off_jn
    jn.children_names = [j2]
    jn.parent_name = j1

    ja.children_names[ja.children_names.index(jb.name)] = jn.name
    jb.offset = off_jb
    jb.parent_name = jn.name
    obj.offset_data[new_name] = jn

    # ---- motion data ---- #
    mb: JointMotion = obj.motion_data[j2]
    mn = deepcopy(mb)
    mn.name = new_name
    mn.data = [[0 for _ in range(len(mb.data[0]))] for _ in range(len(mb.data))]  # no motion!
    obj.motion_data[new_name] = mn

    # ---- reorder ---- #
    return reorder_bvh(obj)


def zero_motion(obj: BVH, name: str):
    """
    :param obj:
    :param name:
    :return:
    """
    data = obj.motion_data[name].data
    for i in range(len(data)):
        data[i] = [0, 0, 0]

    return obj


def append_joint(obj: BVH, parent_name: str, new_name: str, offset):
    """
    :param obj:
    :param parent_name:
    :param new_name:
    :param offset:
    :return:
    """
    pn = parent_name
    pj: JointOffset = obj.offset_data[pn]
    pm: JointMotion = obj.motion_data[pn]
    pj.children_names.append(new_name)

    # offset
    nj = deepcopy(pj)
    nj.parent_name = pn
    nj.children_names = []
    nj.offset = deepcopy(offset)
    nj.name = new_name
    obj.offset_data[new_name] = nj

    # motion
    nm = deepcopy(pm)
    nm.name = new_name
    nm.data = [[0 for _ in range(len(pm.data[0]))] for _ in range(len(pm.data))]  # no motion!
    obj.motion_data[new_name] = nm

    # ---- reorder ---- #
    return reorder_bvh(obj)


def rename_joints(obj: BVH, src_names: list, dst_names: list):
    def __find_and_replace(name_):
        index_ = src_names.index(name_)
        new_ = dst_names[index_]

        # parent
        if name_ != obj.root_name:
            p_name = obj.offset_data[name_].parent_name
            obj.offset_data[p_name].children_names.remove(name_)
            obj.offset_data[p_name].children_names.append(new_)

        # children
        for cn in obj.offset_data[name_].children_names:
            obj.offset_data[cn].parent_name = new_

        # self
        obj.offset_data[name_].name = new_
        obj.motion_data[name_].name = new_

        jo = obj.offset_data[name_]
        jm = obj.motion_data[name_]

        del obj.offset_data[name_]
        del obj.motion_data[name_]

        obj.offset_data[new_] = jo
        obj.motion_data[new_] = jm

    nm_list = [name for name, _ in obj.dfs()][::-1]

    for name in nm_list:
        if name in src_names:
            __find_and_replace(name)

    return reorder_bvh(obj)


def copy_rotations_by_name(src_bvh: BVH, dst_bvh: BVH, dst2src_mapping: dict) -> BVH:
    ret_bvh = deepcopy(dst_bvh)
    ret_bvh.motion_data = OrderedDict()

    ret_bvh.frames = src_bvh.frames
    ret_bvh.frame_time = src_bvh.frame_time
    ret_bvh.filepath = ""

    dst_names = dst_bvh.names

    for dst_name in dst_names:
        if dst_name in dst2src_mapping:
            jm: JointMotion = src_bvh.motion_data[dst2src_mapping[dst_name]]
        else:
            jm: JointMotion = JointMotion(dst_name, [[0.0, 0.0, 0.0] for _ in range(ret_bvh.frames)])

        ret_bvh.motion_data[dst_name] = deepcopy(jm)

    root = ret_bvh.root_name
    ret_bvh.motion_data[root].data = [[0, 0, 0, e[3], e[4], e[5]] for e in ret_bvh.motion_data[root].data]
    return ret_bvh


def retarget(src_bvh: BVH, dst_bvh: BVH, dst_feet: List[str], 
             dst_to_src: Dict[str, str],
             src_t=None, dst_t=None, 
             dst_head: str=None,
             get_dst_off_from_t=False,
             foot_ik=True) -> BVH:
    """retarget a motion from src_bvh to dst_bvh, given a full joint mapping from dst to src

    Args:
        src_bvh (BVH): source bvh
        dst_bvh (BVH): destination bvh
        dst_feet (List[str]): feet's name for foot ik (place on ground)
        dst_to_src (Dict[str, str]): joint name mapping
        src_t (torch.Tensor, optional): some motion may not in T-pose (e.g. A-posed), pass a [J, 3, T] tensor if necessary. Defaults to None.
        dst_t (torch.Tensor, optional): some motion may not in T-pose (e.g. A-posed), pass a [J, 3, T] tensor if necessary. Defaults to None.

    Raises:
        KeyError: joint name mapping mismatch

    Returns:
        BVH: retargeted bvh motion file
    """
    from ..motion_tensor.bvh_casting import get_positions_from_bvh, get_t_pose_from_bvh, write_quaternion_to_bvh, get_offsets_from_bvh, write_offsets_to_bvh
    from ..motion_tensor.rotations import get_quat_from_pos, quaternion_to_matrix
    from ..motion_tensor.motion_process import get_feet_contacts, get_feet_grounding_shift
    from ..motion_tensor.kinematics import forward_kinematics

    if not isinstance(src_bvh, BVH):
        src_bvh = BVH(src_bvh)

    if not isinstance(dst_bvh, BVH):
        dst_bvh = BVH(dst_bvh)

    if src_t is None:
        src_t = get_t_pose_from_bvh(src_bvh)
    if dst_t is None:
        dst_t = get_t_pose_from_bvh(dst_bvh)

    # frames = src_pos.shape[-1]
    src_pdx = src_bvh.p_index
    dst_pdx = dst_bvh.p_index
    src_names = src_bvh.names
    dst_names = dst_bvh.names

    src_head = dst_to_src[dst_head]
    src_head_id = src_names.index(src_head)
    dst_head_id = dst_names.index(dst_head)
    src_feet_id = [src_names.index(dst_to_src[e]) for e in dst_feet]
    dst_feet_id = [dst_names.index(e)             for e in dst_feet]

    if dst_head is None:
        dst_x = (dst_t[:, 0, :].max() - dst_t[:, 0, :].min()).item()
        dst_y = (dst_t[:, 1, :].max() - dst_t[:, 1, :].min()).item()
        dst_z = (dst_t[:, 2, :].max() - dst_t[:, 2, :].min()).item()
    else:
        dst_x = max(abs(dst_t[dst_head_id, 0, :] - dst_t[feet_id, 0, :]) for feet_id in dst_feet_id)
        dst_y = max(abs(dst_t[dst_head_id, 1, :] - dst_t[feet_id, 1, :]) for feet_id in dst_feet_id)
        dst_z = max(abs(dst_t[dst_head_id, 2, :] - dst_t[feet_id, 2, :]) for feet_id in dst_feet_id)

    if src_head is None:
        src_x = (src_t[:, 0, :].max() - src_t[:, 0, :].min()).item()
        src_y = (src_t[:, 1, :].max() - src_t[:, 1, :].min()).item()
        src_z = (src_t[:, 2, :].max() - src_t[:, 2, :].min()).item()
    else:
        src_x = max(abs(src_t[src_head_id, 0, :] - src_t[feet_id, 0, :]) for feet_id in src_feet_id)
        src_y = max(abs(src_t[src_head_id, 1, :] - src_t[feet_id, 1, :]) for feet_id in src_feet_id)
        src_z = max(abs(src_t[src_head_id, 2, :] - src_t[feet_id, 2, :]) for feet_id in src_feet_id)

    src_h = max(src_x, src_y, src_z)
    dst_h = max(dst_x, dst_y, dst_z)

    src_pos, src_off, src_trs, src_qua = get_positions_from_bvh(src_bvh, return_rest=True)
    # dst_pos, dst_off, dst_trs, dst_qua = get_positions_from_bvh(dst_bvh, return_rest=True)

    if get_dst_off_from_t is True:
        dst_off = torch.clone(dst_t)
        for p, c in zip(dst_pdx[::-1], list(range(len(dst_pdx)))[::-1]):
            if p >= 0:
                dst_off[c] -= dst_off[p].clone()
    else:
        dst_off = get_offsets_from_bvh(dst_bvh)

    src_names_mapping = []
    for d_name in dst_names:
        s_name = dst_to_src.get(d_name, None)
        if s_name is None:
            raise KeyError(f"Name {d_name} not in dst_to_src mapping.")
        src_names_mapping.append(s_name)

    src_map = [src_names.index(e) for e in src_names_mapping]
    # src_map = [src_names.index(e) for e in dst2src_mapping.values()]
    # dst_map = [dst_names.index(e) for e in dst2src_mapping.keys()]

    dst_pos = torch.clone(src_pos[src_map])
    dst_pos = dst_pos * (dst_h / src_h)

    # # >>> DEBUG
    # def offset_to_position(pdx, offset):
    #     offset = torch.clone(offset)
    #     for i, p in enumerate(pdx):
    #         if p == -1:
    #             continue
    #         offset[i, ...] += offset[p, ...]
    #     return offset
    
    # from fmbvh.visualization.cvrnd import render_pose
    # DEBUG_N = len(dst_pdx)
    # render_pose(dst_pdx[:DEBUG_N],                              dst_pos[:DEBUG_N], None, scale=1.0)
    # render_pose(dst_pdx[:DEBUG_N], offset_to_position(dst_pdx, dst_off)[:DEBUG_N], None, scale=1.0)
    # # <<< DEBUG

    dst_qua = torch.clone(src_qua[src_map])
    dst_trs = src_trs * (dst_h / src_h)

    dst_trs, dst_qua = get_quat_from_pos(dst_pdx, dst_pos, dst_off, "kabsch")

    # # >>> DEBUG
    # # dst_off = src_off[src_map].clone()
    # dst_fk_pos = forward_kinematics(dst_pdx, quaternion_to_matrix(dst_qua), dst_trs, dst_off)
    # from fmbvh.visualization.cvrnd import render_pose
    # render_pose(dst_pdx, dst_fk_pos, None, scale=1.0)
    # exit()
    # # DEBUG <<<

    # fix foot
    if foot_ik:
        K = int(1+2*((src_bvh.fps+0.5)//15))
        fc = get_feet_contacts(src_pos, src_feet_id, src_h, kernel_size=0)
        dst_pos = forward_kinematics(dst_pdx, quaternion_to_matrix(dst_qua), dst_trs, dst_off)
        sft = get_feet_grounding_shift(dst_pos[dst_feet_id], fc, up_axis=1, kernel=K+4, iter_=5, gather="min")
        dst_trs[0, 1] -= sft

    ret_bvh = deepcopy(dst_bvh)
    if get_dst_off_from_t:
        ret_bvh = write_offsets_to_bvh(dst_off, ret_bvh)
    ret_bvh.filepath = ""
    return write_quaternion_to_bvh(dst_trs, dst_qua, ret_bvh, frame_time=src_bvh.frame_time)


def retarget_cr(src_bvh: BVH, dst_bvh: BVH, dst_feet: List[str], 
                dst_to_src: Dict[str, str],
                src_t=None, dst_t=None, 
                dst_head: str=None,
                get_dst_off_from_t=True,
                foot_placement=False) -> BVH:
    """retarget a motion from src_bvh to dst_bvh by copy rotations, given a full joint mapping from dst to src

    Args:
        src_bvh (BVH): source bvh
        dst_bvh (BVH): destination bvh
        dst_feet (List[str]): feet's name for foot ik (place on ground)
        dst_to_src (Dict[str, str]): joint name mapping
        src_t (torch.Tensor, optional): some motion may not in T-pose (e.g. A-posed), pass a [J, 3, T] tensor if necessary. Defaults to None.
        dst_t (torch.Tensor, optional): some motion may not in T-pose (e.g. A-posed), pass a [J, 3, T] tensor if necessary. Defaults to None.

    Raises:
        KeyError: joint name mapping mismatch

    Returns:
        BVH: retargeted bvh motion file
    """
    from ..motion_tensor.bvh_casting import get_quaternion_from_bvh, write_quaternion_to_bvh, get_t_pose_from_bvh, get_offsets_from_bvh, write_offsets_to_bvh, get_positions_from_bvh
    from ..motion_tensor.rotations import get_quat_from_pos, quaternion_to_matrix, quaternion_from_two_vectors, mul_two_quaternions, inverse_quaternion
    from ..motion_tensor.motion_process import get_feet_contacts, get_feet_grounding_shift
    from ..motion_tensor.kinematics import forward_kinematics

    if not isinstance(src_bvh, BVH):
        src_bvh = BVH(src_bvh)

    if not isinstance(dst_bvh, BVH):
        dst_bvh = BVH(dst_bvh)

    if src_t is None:
        src_t = get_t_pose_from_bvh(src_bvh)
    if dst_t is None:
        dst_t = get_t_pose_from_bvh(dst_bvh)

    # frames = src_pos.shape[-1]
    src_pdx = src_bvh.p_index
    dst_pdx = dst_bvh.p_index
    src_names = src_bvh.names
    dst_names = dst_bvh.names

    src_head = dst_to_src[dst_head]
    src_head_id = src_names.index(src_head)
    dst_head_id = dst_names.index(dst_head)
    src_feet_id = [src_names.index(dst_to_src[e]) for e in dst_feet]
    dst_feet_id = [dst_names.index(e)             for e in dst_feet]

    if dst_head is None:
        dst_x = (dst_t[:, 0, :].max() - dst_t[:, 0, :].min()).item()
        dst_y = (dst_t[:, 1, :].max() - dst_t[:, 1, :].min()).item()
        dst_z = (dst_t[:, 2, :].max() - dst_t[:, 2, :].min()).item()
    else:
        dst_x = max(abs(dst_t[dst_head_id, 0, :] - dst_t[feet_id, 0, :]) for feet_id in dst_feet_id)
        dst_y = max(abs(dst_t[dst_head_id, 1, :] - dst_t[feet_id, 1, :]) for feet_id in dst_feet_id)
        dst_z = max(abs(dst_t[dst_head_id, 2, :] - dst_t[feet_id, 2, :]) for feet_id in dst_feet_id)

    if src_head is None:
        src_x = (src_t[:, 0, :].max() - src_t[:, 0, :].min()).item()
        src_y = (src_t[:, 1, :].max() - src_t[:, 1, :].min()).item()
        src_z = (src_t[:, 2, :].max() - src_t[:, 2, :].min()).item()
    else:
        src_x = max(abs(src_t[src_head_id, 0, :] - src_t[feet_id, 0, :]) for feet_id in src_feet_id)
        src_y = max(abs(src_t[src_head_id, 1, :] - src_t[feet_id, 1, :]) for feet_id in src_feet_id)
        src_z = max(abs(src_t[src_head_id, 2, :] - src_t[feet_id, 2, :]) for feet_id in src_feet_id)

    src_h = max(src_x, src_y, src_z)
    dst_h = max(dst_x, dst_y, dst_z)

    src_trs, src_qua = get_quaternion_from_bvh(src_bvh)

    src_names_mapping = []
    for d_name in dst_names:
        s_name = dst_to_src.get(d_name, None)
        if s_name is None:
            raise KeyError(f"Name {d_name} not in dst_to_src mapping.")
        src_names_mapping.append(s_name)
    src_map = [src_names.index(e) for e in src_names_mapping]

    if get_dst_off_from_t is True:
        dst_off = torch.clone(dst_t)
        for p, c in zip(dst_pdx[::-1], list(range(len(dst_pdx)))[::-1]):
            if p >= 0:
                dst_off[c] -= dst_off[p].clone()
    else:
        dst_off = get_offsets_from_bvh(dst_bvh)

    dst_qua = torch.clone(src_qua[src_map])
    dst_trs = src_trs * (dst_h / src_h)

    tgt_t = torch.clone(src_t[src_map])
    _, delta_qua = get_quat_from_pos(dst_pdx, tgt_t, dst_off, "kabsch")

    c_global = delta_qua.clone()
    for c, p in enumerate(dst_pdx):
        if p >= 0:
            c_global[c] = mul_two_quaternions(c_global[p], c_global[c])

    p_global = delta_qua.clone()
    for c, p in enumerate(dst_pdx):
        if p >= 0:
            p_global[c] = c_global[p]
        else:
            p_global[c, :, :] = 0
            p_global[c, 0, :] = 1

    dst_qua = mul_two_quaternions(inverse_quaternion(p_global), mul_two_quaternions(dst_qua, c_global))

    # fix foot
    if foot_placement:
        src_pos = get_positions_from_bvh(src_bvh)
        K = int(1+2*((src_bvh.fps+0.5)//15))
        fc = get_feet_contacts(src_pos, src_feet_id, src_h, kernel_size=0)
        dst_pos = forward_kinematics(dst_pdx, quaternion_to_matrix(dst_qua), dst_trs, dst_off)
        sft = get_feet_grounding_shift(dst_pos[dst_feet_id], fc, up_axis=1, kernel=K+4, iter_=5, gather="max")
        dst_trs[0, 1] -= sft

    ret_bvh = deepcopy(dst_bvh)
    if get_dst_off_from_t:
        ret_bvh = write_offsets_to_bvh(dst_off, ret_bvh)
    ret_bvh.filepath = ""
    return write_quaternion_to_bvh(dst_trs, dst_qua, ret_bvh, frame_time=src_bvh.frame_time)


def uneven_ground_to_plane(dst_bvh: BVH, dst_feet: List[str], vel_thres=0.005, pos_thres=0.03):
    from ..motion_tensor.motion_process import get_feet_contacts, get_feet_grounding_shift
    from ..motion_tensor.bvh_casting import get_positions_from_bvh, get_t_pose_from_bvh, write_quaternion_to_bvh, get_offsets_from_bvh
    from ..motion_tensor.rotations import get_quat_from_pos, quaternion_to_matrix
    from ..motion_tensor.motion_process import get_feet_contacts, get_feet_grounding_shift
    from ..motion_tensor.kinematics import forward_kinematics

    dst_names = dst_bvh.names
    dst_pdx = dst_bvh.p_index
    dst_feet_id = [dst_names.index(e)             for e in dst_feet]

    dst_pos, dst_off, dst_trs, dst_qua = get_positions_from_bvh(dst_bvh, return_rest=True)
    dst_t = get_t_pose_from_bvh(dst_bvh)

    dst_x = (dst_t[:, 0, :].max() - dst_t[:, 0, :].min()).item()
    dst_y = (dst_t[:, 1, :].max() - dst_t[:, 1, :].min()).item()
    dst_z = (dst_t[:, 2, :].max() - dst_t[:, 2, :].min()).item()

    dst_h = max(dst_x, dst_y, dst_z)

    K = int(1+2*((dst_bvh.fps+0.5)//15))
    fc_p = get_feet_contacts(dst_pos, dst_feet_id, dst_h, kernel_size=K+2, criteria='pos', pos_thres=pos_thres)
    fc_v = get_feet_contacts(dst_pos, dst_feet_id, dst_h, kernel_size=K+2, criteria='vel', vel_thres=vel_thres)

    fc = torch.max(fc_p, fc_v)  # [..., E, T]

    ft_sft = get_feet_grounding_shift(dst_pos[dst_feet_id], fc, up_axis=1, kernel=K+8, iter_=5, gather="none")

    # root_sft = ft_sft.max(dim=0)[0]
    root_sft = ft_sft.mean(dim=0)

    ft_sft += root_sft[None, ...]

    dst_trs[:, 1, :] -= root_sft
    dst_pos[dst_feet_id, 1, :] -= ft_sft

    dst_trs, dst_qua = get_quat_from_pos(dst_pdx, dst_pos, dst_off, "kabsch")

    return write_quaternion_to_bvh(dst_trs, dst_qua, dst_bvh)


# def demo_test():
#     show_t_pose = False
#     is_cmu = True
#     edge_repr = True
#
#     if not is_cmu:
#         bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Ortiz_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Jasper_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Abe_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Malcolm_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Knight_m.bvh')
#
#         remain_names = [
#             'Hips',
#             'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head', 'HeadTop_End',
#             'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand',
#             'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand',
#             'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End',
#             'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End'
#         ]
#         remove_names = get_remaining_joint_names(bvh_obj, remain_names)
#         cut_off_joints(bvh_obj, remove_names)
#     else:
#         bvh_obj = BVH(r'D:\_dataset\Motion Style Transfer\mocap_xia\0_angry\angry_01_000.bvh')
#         insert_joint_between(bvh_obj, 'LowerBack', 'Spine', 'MySpine')
#         delete_joints(bvh_obj, 'LowerBack')
#         insert_joint_between(bvh_obj, 'Spine', 'Spine1', 'MySpine2')
#
#         rectify_joint(bvh_obj, 'LeftUpLeg', 'LeftLeg', [0, -1, 0])
#         rectify_joint(bvh_obj, 'RightUpLeg', 'RightLeg', [0, -1, 0])
#
#     from visualization.visualize_motion import MoVisualizer
#     import motion_tensor as mot
#     import torch
#
#     t_pos = mot.bvh_casting.get_positions_from_bvh(bvh_obj)[..., 0]  # [J, 3]
#     max_y = torch.max(t_pos[:, 1])
#     min_y = torch.min(t_pos[:, 1])
#     height = (max_y - min_y).item()
#
#     trs, qua = mot.bvh_casting.get_quaternion_from_bvh(bvh_obj)  # [1, 3, F], [J, 4, F]
#     trs /= height
#
#     if show_t_pose:
#         trs[:, :, :] = 0.0
#         qua[:, :1, :] = 1.0
#         qua[:, 1:, :] = 0.0
#
#     mat = mot.rotations.quaternion_to_matrix(qua[None, ...])  # [B, J, 3, 3, F]
#     J, _, F = qua.shape
#
#     offsets = mot.bvh_casting.get_offsets_from_bvh(bvh_obj)[None, ...]  # [B, J, 3, 1]
#     offsets = torch.broadcast_to(offsets, (1, J, 3, F))  # [B, J, 3, F]
#
#     if not edge_repr:
#         fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
#                                                    trs[None, ...], offsets, is_edge=False)    # [B, J, 3, F]
#     else:
#         ide = torch.eye(3)[None, None, ..., None]
#         ide = torch.broadcast_to(ide, (1, 1, 3, 3, F))
#         edge_i = [e for e in bvh_obj.dfs_parent() if e != -1]
#         mat = mat[:, edge_i, :, :, :]
#         mat = torch.concat([ide, mat], dim=1)  # append root rotation
#         fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
#                                                    trs[None, ...], offsets, is_edge=True)    # [B, J, 3, F]
#
#     fk_pos /= height
#
#     def _next():
#         f = 0
#         while True:
#             rpt = trs[:, :, f]  # [1, 3]
#             cps = fk_pos[0, :, :, f]  # [J, 3]
#             pos_ls = cps + rpt  # [J, 3]
#             yield pos_ls.numpy().tolist()
#             f = (f + 1) % F
#
#     p_index = bvh_obj.dfs_parent()
#     mvz = MoVisualizer(p_index, _next(), max_fps=30, add_coordinate=True)
#     # mvz.add_grids(10, 10, height*0.2)
#     mvz.run()
#
#
# if __name__ == '__main__':
#     demo_test()
