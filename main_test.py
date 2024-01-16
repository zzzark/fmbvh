import numpy as np
from visualization.utils import quick_visualize, quick_visualize_fk
from motion_tensor.rotations import quaternion_from_two_vectors as v2q
from motion_tensor.rotations import pad_position_to_quaternion as pad
from motion_tensor.rotations import mul_two_quaternions as mul
from motion_tensor.rotations import inverse_quaternion as inv
from motion_tensor.rotations import conjugate_quaternion as cnjc
from motion_tensor.rotations import normalize_quaternion as norm
from motion_tensor.rotations import quaternion_to_matrix as q2m
from motion_tensor.rotations import norm_of_quaternion as nof
from motion_tensor.rotations import euler_to_quaternion as e2q
from motion_tensor.rotations import slerp_n


import json
import torch


def np_array_to_json(path, trans, quats, w_first=False):
    """
    :param path: 
    :param trans: [F, 3]
    :param quats: [F, J, 4]
    :return: 
    """
    assert isinstance(trans, np.ndarray)
    assert isinstance(quats, np.ndarray)

    def qua4(e):
        assert len(e) == 4
        if w_first:
            return {'w': e[0], 'x': e[1], 'y': e[2], 'z': e[3]}
        else:
            return {'x': e[0], 'y': e[1], 'z': e[2], 'w': e[3]}

    def vec3(e):
        assert len(e) == 3
        return {'x': e[0], 'y': e[1], 'z': e[2]}

    with open(path, 'w') as json_file:
        F = trans.shape[0]
        trans = trans.tolist()
        quats = quats.tolist()
        content = ""
        for i in range(F):
            p = vec3(trans[i])
            q = [qua4(e) for e in quats[i]]
            content += json.dumps({"p": p, "q": q}) + "\n"
        json_file.write(content)


def to_np(*x):
    return tuple([e.numpy() for e in x]) if len(x) > 1 else x[0].numpy()


def to_th(*x):
    return tuple([torch.from_numpy(e) for e in x]) if len(x) > 1 else torch.from_numpy(x[0])


def rotate_at(q, p, indices=None):
    """
    :param q: [J, 4, F]
    :param p: [J, 3, F]
    :param indices: index, int or array
    :return: [J, 3, F]
    """
    if isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = slice(None, None, None)
    p = p.clone()
    old_p = p
    p = pad(p[indices])
    r = mul(q, mul(p, inv(q)))
    r = r[:, 1:, :]

    old_p[indices] = r
    return old_p


def get_children(p_index, cur_index) -> list:
    """
    get kinematic chain (all children of cur_index excludes cur_index)
    """
    chain_list = {cur_index}
    for c, p in enumerate(p_index):
        if p in chain_list:
            chain_list.add(c)
    chain_list.remove(cur_index)
    return list(chain_list)


def get_rotation_at(offset, target, p_index, i):
    """
    :param p1: [J, 3, F]
    :param p2: [J, 3, F]
    :param p_index:
    :param i:
    :return: [1, 4, F]
    """
    children = [e for e in range(len(p_index)) if p_index[e] == i]

    if len(children) == 0:
        ret = torch.zeros(1, 4, target.shape[-1], dtype=target.dtype)
        ret[:, 0, :] = 1.0
        return ret

    r_ls = []
    for c in children:
        a = offset[[c], :, :].broadcast_to(1, 3, target.shape[-1])
        b = target[[c], :, :]

        r = v2q(a, b)
        r_ls.append(r)

    r = slerp_n(*r_ls)
    return r


def main():
    t_json = """
    [[[0, 0, 0],
    [-0.0561437, -0.09454167, -0.02347454],
    [0.05786965, -0.1051669, -0.01655883],
    [0.001336131, 0.1104168, -0.03792468],
    [-0.06722913, -0.3968683, -0.006654377],
    [0.0507268, -0.3798212, -0.01445162],
    [-0.01020924, 0.1509713, 0.00444234],
    [0.04559392, -0.4212858, -0.04114642],
    [-0.01715488, -0.434912, -0.03993758],
    [0.008991977, 0.05784686, 0.0226697],
    [-0.04430889, -0.06073571, 0.1351566],
    [0.03544413, -0.05966973, 0.1418128],
    [0.009582404, 0.1660209, -0.02724041],
    [-0.04242062, 0.07627739, -0.005254913],
    [0.0460769, 0.07681628, -0.008529007],
    [-0.02290352, 0.1607197, 0.02293979],
    [-0.1409457, 0.06041686, -0.01489903],
    [0.1303553, 0.05884924, -0.01258629],
    [-0.2547487, -0.07616559, -0.04549227],
    [0.2724926, -0.0459256, -0.03242894],
    [-0.2716039, 0.0256131, -0.000648317],
    [0.2633793, 2.866859E-05, -0.01182221]]]
    """

    p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    # mirrored = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]

    # ------
    with open("X:/all_pos.json", "r") as f:
        target = json.load(f)
    target = np.array(target).astype(float)
    target = np.transpose(target, (1, 2, 0))  # [F, J, 3] -> [J, 3, F]
    
    # # ------ DEBUG DATA ------ #
    # t_json = """
    # [[[0, 0, 0.0],
    #   [1, 0, 0.0],
    #   [1, 0, 0.0],
    #   [1, 0, 0.0]]]
    # """
    # p_index = [-1, 0, 1, 2]
    # mirrored = []
    # n_json = """
    # [[[ 0, 0, 0],
    #   [ 0, 1, 0],
    #   [-1, 1, 0],
    #   [-1, 0, 0]]]
    # """
    # np_obj = json.loads(n_json)
    # np_obj = np.array(np_obj).astype(float)
    # np_obj = np.transpose(np_obj, (1, 2, 0))  # [F, J, 3] -> [J, 3, F]
    # # ------ DEBUG ------ #

    offset = json.loads(t_json)
    offset = np.array(offset)  # [1, 22, 3]
    offset = np.transpose(offset, (1, 2, 0))  # [F, J, 3] -> [J, 3, F]
    
    # ------
    target, offset = to_th(target, offset)

    # ---- FIXME: reverse x-axis ---- # 
    target[:, 0, :].neg_()

    target_trs = target[[0], :, :].clone()
    target[:, :, :] -= target_trs

    quick_visualize(p_index, target + target_trs, 2.0)

    quas = []
    for c, p in enumerate(p_index):
        children = get_children(p_index, c)
        if p >= 0:  # assert p_offset == (0, 0, 0)
            target[children] -= offset[[c]]
        q = get_rotation_at(offset, target, p_index, c)
        target = rotate_at(inv(q), target, children)
        quas.append(q)
    
    quas = torch.concatenate(quas, dim=0)

    # ---- to json ---- #
    trans = target_trs[0].permute(1, 0).numpy()
    quats = quas.permute(2, 0, 1).numpy()
    np_array_to_json("X:/fwdq.txt", trans, quats, w_first=True)
    # ----------------- # 

    quick_visualize_fk(p_index, offset, quas, target_trs, 2.0)


main()


