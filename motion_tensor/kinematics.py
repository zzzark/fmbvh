import torch
from .rotations import quaternion_to_matrix
from .rotations import quaternion_from_two_vectors
from .rotations import quaternion_rotate_vector_inv
from .rotations import mul_two_quaternions
from .motion_process import get_feet_contact_points
from .motion_process import get_feet_contacts, get_feet_grounding_shift


import torch
from typing import Tuple


def forward_kinematics(parent_index: list, mat3x3: torch.Tensor,
                       root_pos: Tuple[torch.Tensor, None], offset: torch.Tensor,
                       world=True, is_edge=False):
    """
    implement forward kinematics in a batched manner
    :param parent_index: index of parents (-1 for no parent)
    :param mat3x3: rotation matrix, [(B), J, 3, 3, F] (batch_size x joint_num x 3 x 3 x frame_num)
    :param root_pos: root position [(B), 1, 3, F], None for Zero
    :param offset: joint offsets [(B), J, 3, F]
    :param world: world position or local position ?
    :param is_edge:
            True:  mat3x3[i] represents a rotation matrix of the parent of joint i,
                   i.e. edge rotation
            False: mat3x3[i] represents a rotation matrix of joint i,
                   i.e. joint rotation
    :return: tensor of positions in the shape of [(B), J, 3, F]
    """
    assert parent_index[0] == -1, f"the first parent index should be -1 (root), not {parent_index[0]}."

    batch = len(mat3x3.shape) == 5
    if not batch:
        mat3x3 = mat3x3[None, ...]
        if root_pos is not None:
            root_pos = root_pos[None, ...]
        offset = offset[None, ...]

    assert len(mat3x3.shape) == 5
    assert len(offset.shape) == 4
    assert root_pos is None or len(root_pos.shape) == 4

    B, J, _, _, F = mat3x3.shape

    mat3x3 = mat3x3.permute(0, 4, 1, 2, 3)                  # mat:    [B, F, J, 3, 3]
    offset = offset.permute(0, 3, 1, 2)[..., None]          # offset: [B, F, J, 3, 1]
    if root_pos is not None:
        root_pos = root_pos.permute(0, 3, 1, 2)[..., None]  # root:   [B, F, 1, 3, 1]

    mat_mix = torch.empty_like(mat3x3, dtype=mat3x3.dtype, device=mat3x3.device)  # avoid in-place operation

    position = torch.empty((B, F, J, 3, 1), device=offset.device)  # [B, F, J, 3, 1]

    if root_pos is not None:
        position[..., 0, :, :] = root_pos[..., 0, :, :]
    else:
        position[..., 0, :, :].zero_()

    mat_mix[..., 0, :, :] = mat3x3[..., 0, :, :]
    for ci, pi in enumerate(parent_index[1:], 1):
        off_i = offset[..., ci, :, :]

        if not is_edge:
            mat_p = mat_mix[..., pi, :, :]
            trs_i = torch.matmul(mat_p, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = torch.matmul(mat_p, mat3x3[..., ci, :, :])
        else:
            combo = torch.matmul(mat_mix[..., pi, :, :], mat3x3[..., ci, :, :])
            trs_i = torch.matmul(combo, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = combo

        if world:
            position[..., ci, :, :] += position[..., pi, :, :]

    position = position[..., 0].permute(0, 2, 3, 1)

    if not batch:
        position = position[0]

    return position


def inverse_kinematics_grad(p_index, off, trs, qua, tg_pos, ee_ids, height, silence=False,
                            iteration=50, vel_mul=0.2, qua_mul=0.01):
    """
    :param p_index:
    :param off: bone offsets
    :param trs: original root trs (update inplace)
    :param qua: original full qua (update inplace)
    :param tg_pos: full body target position
    :param ee_ids:
    :param height:
    :param silence:
    :param iteration:
    :param vel_mul: velocity constraint multiplier in loss function
    :param qua_mul: quaternion constraint multiplier in loss function
    :return:
    """
    tg_ee_pos = tg_pos[ee_ids]
    trs = trs.clone()
    qua = qua.clone()

    org_qua = qua.clone()

    trs.requires_grad = True
    qua.requires_grad = True

    # optim = torch.optim.Adam([trs, qua], lr=0.001, betas=(0.9, 0.9))
    optim = torch.optim.SGD([trs, qua], lr=0.003, momentum=0.9)
    l2 = torch.nn.MSELoss()

    for t in range(iteration):
        optim.zero_grad()

        m = quaternion_to_matrix(qua)
        ik_pos = forward_kinematics(p_index, m, trs, off, True, False)
        ik_ee_pos = ik_pos[ee_ids]

        ik_vel = ik_pos[..., :-1] - ik_pos[..., 1:]
        tg_vel = tg_pos[..., :-1] - tg_pos[..., 1:]
        ee_loss = l2(ik_ee_pos, tg_ee_pos)
        other_loss = vel_mul * l2(ik_vel, tg_vel) + qua_mul * height * l2(qua, org_qua)
        loss = ee_loss + other_loss

        loss.backward()
        optim.step()

        if not silence:
            print(f"ik loss {t+1}/{iteration}: {ee_loss.item() / height}")

    trs.requires_grad = False
    qua.requires_grad = False
    return trs, qua


def inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ee_ids, height, sin_lim=None,
                              silence=False, iteration=10, return_pos=False):
    # TODO: fix bugs
    
    """
    :param p_index:
    :param off: bone offsets
    :param trs: original root trs (update inplace)
    :param qua: original full qua (update inplace)
    :param tg_pos: full body target position
    :param ee_ids:
    :param height:
    :param sin_lim: tuple or list of tuple, rotation angle limit (sine) for each ee.
                    positive: foot base above toe; negative: foot base below toe
    :param silence:
    :param iteration:
    :param return_pos: return the target pos after ik
    :return: (trs, qua) or (trs, qua, pos)
    """
    
    if isinstance(sin_lim ,tuple):
        raise NotImplementedError
    if isinstance(sin_lim, tuple) and isinstance(sin_lim[0], float):
        sin_lim = [sin_lim for _ in range(len(ee_ids))]

    # og_pos = forward_kinematics(p_index, qua, trs, off, True, False)
    # ik_pos = og_pos.clone()

    mat = quaternion_to_matrix(qua)
    ik_pos = forward_kinematics(p_index, mat, trs, off, True, False)
    trs = trs.clone()
    qua = qua.clone()

    def __bone_len(bn):
        return (((bn ** 2.0).sum(dim=-2)) ** 0.5)

    for t in range(iteration):
        for i, ee in enumerate(ee_ids):
            j = ee
            kin_chain = []  # kinematic chain
            while j > 0:  # TODO: take sub-base cases into consideration (e.g. hand base)
                kin_chain.append(j)
                j = p_index[j]

            # forward
            tg_j = tg_pos[ee].clone()
            for jc, jp in zip(kin_chain[:-1], kin_chain[1:]):  # (child, parent)
                if jc == ee:
                    d = ik_pos[jp] - ik_pos[jc]  # FIXME: direction is at the fixed direction
                    ik_pos[jc] = tg_j.clone()
                    len_off = torch.linalg.vector_norm(off[jc], dim=0).item()

                    # FIXME: the following code is only dedicated for handling feet
                    #        (it constraints the angle by simply compute the according y-dim value)
                    #        --
                    #        for removing foot sliding artifacts, sin_lim should be a negative value
                    #        for adapting to terrain height, sin_lim can be positive
                    #        --
                    #        should move the code below to function `get_ik_target_pos`
                    #        and pass the terrain info to fix foot sliding & terrain adaption issues
                    #        this requires the impl of FABRIK can handle multiple joint constraints
                    # >>>>
                    if sin_lim is not None:
                        y_min, y_max = len_off * sin_lim[i][0], len_off * sin_lim[i][1]
                        y_d = d[1, :]
                        y_d[y_d < y_min] = y_min
                        y_d[y_d > y_max] = y_max
                        d[1, :] = y_d
                    # <<<<
                    d /= torch.linalg.vector_norm(d, dim=0)
                    d *= len_off
                else:
                    ik_pos[jc] = tg_j.clone()
                    d = ik_pos[jp] - ik_pos[jc]
                    d /= torch.linalg.vector_norm(d, dim=0)
                    len_off = torch.linalg.vector_norm(off[jc], dim=0).item()
                    d *= len_off

                tg_j += d

            # ... move root to target point && move back ...

            # backward
            kin_chain = kin_chain[::-1]
            tg_j = ik_pos[kin_chain[0]].clone()
            for jp, jc in zip(kin_chain[:-1], kin_chain[1:]):
                ik_pos[jp] = tg_j.clone()
                d = ik_pos[jc] - ik_pos[jp]

                len_off = torch.linalg.vector_norm(off[jc], dim=0).item()
                d /= torch.linalg.vector_norm(d, dim=0)
                d *= len_off
                tg_j += d

            # last joint
            ik_pos[ee] = tg_j

        if not silence:
            ee_loss = ((tg_pos[ee_ids] - ik_pos[ee_ids])**2.0).mean()  # MSELoss
            print(f"ik loss {t+1}/{iteration}: {ee_loss.item() / height}")

    # solve the joint rotations
    for ee in ee_ids:
        j = ee
        kin_chain = []  # kinematic chain
        while j >= 0:
            kin_chain.append(j)
            j = p_index[j]
        kin_chain = kin_chain[::-1]

        # id quaternion
        last_q = qua[kin_chain[0]]

        for jc, jp in zip(kin_chain[2:], kin_chain[1:]):
            v0 = off[jc]
            v1 = ik_pos[jc] - ik_pos[jp]
            v1 = quaternion_rotate_vector_inv(last_q, v1)  # last_q-1 * v * last_q
            qp = quaternion_from_two_vectors(v0, v1)
            qua[jp] = qp
            last_q = mul_two_quaternions(last_q, qp)

    return (trs, qua, ik_pos) if return_pos else (trs, qua)


def get_ik_target_pos(fc, org_pos, ee_ids, force_on_floor=True, interp_length=13):
    """
    :param fc: (target) foot contact labels  [E, T]
    :param org_pos: original full body position  [J, 3, T]
    :param ee_ids: list
    :param force_on_floor:
    :param interp_length:
    :return: full body target position
    """
    tg_ee_pos = get_feet_contact_points(org_pos[ee_ids], fc, force_on_floor, interp_length)  # ee target position
    tg_pos = org_pos.clone()
    tg_pos[ee_ids] = tg_ee_pos
    return tg_pos


def easy_fix_sliding(p_index, trs, qua, off, hei, ee_ids, ft_ids, up_axis=1, fps=60, **hypers):
    """
    :param p_index: 
    :param trs: [1, 3, T]
    :param qua: [J, 4, T]
    :param off: [J, 3, 1]
    :param ee_ids: indices of hands, feet and head
    :param ft_ids: indices of feet
    :up_axis: 0-x, 1-y, 2-z
    :fps: 
    :hypers: 
    :return: trs, qua
    """
    
    # ---- settings ---- #
    K = int(1+2*((fps+0.5)//15)) if 'K' not in hypers else hypers['K']
    R = 2 if 'R' not in hypers else hypers['R']
    pthres1 = 0.150 if 'pthres1' not in hypers else hypers['pthres1']
    vthres1 = 0.010 if 'vthres1' not in hypers else hypers['vthres1']
    pthres2 = 0.020 if 'pthres2' not in hypers else hypers['pthres2']
    vthres2 = 0.005 if 'vthres2' not in hypers else hypers['vthres2']
    cr1 = "pos" if 'cr1' not in hypers else hypers['cr1']
    cr2 = "vel" if 'cr2' not in hypers else hypers['cr2']

    mat = quaternion_to_matrix(qua)
    pos = forward_kinematics(p_index, mat, trs, off, True, False)

    # DEBUG
    # from ..visualization.utils import quick_visualize as qv, quick_visualize_fk as qfk
    # qv(p_index, pos, 1)

    # -- softly put feet on plane -- # 
    for r in range(R):
        discount = 2 ** (-r/R)
        fp = pos[ft_ids]
        fc = get_feet_contacts(pos, ft_ids, hei, criteria=cr1, pos_thres=pthres1*discount, vel_thres=vthres1*discount, kernel_size=0)
        sft = get_feet_grounding_shift(fp, fc, up_axis=up_axis, kernel=K+4, iter_=5)
        trs[0, up_axis] -= sft
        pos[:, up_axis, :] -= sft

    # get fc
    if True:
        fp = pos[ft_ids]
        fc = get_feet_contacts(pos, ft_ids, hei, criteria=cr2, pos_thres=pthres2, vel_thres=vthres2, kernel_size=K)
        sft = get_feet_grounding_shift(fp, fc, up_axis=up_axis, kernel=K+4, iter_=5)
        trs[0, up_axis] -= sft
        pos[:, up_axis, :] -= sft
    fp = pos[ft_ids]
    fc = get_feet_contacts(pos, ft_ids, hei, criteria=cr2, pos_thres=pthres2, vel_thres=vthres2, kernel_size=K)
    
    # get target pos
    tg_pos = get_ik_target_pos(fc, pos, ft_ids, interp_length=K+4)

    # DEBUG
    # qv(p_index, tg_pos, 1)
    
    ik_trs, ik_qua = inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ft_ids, hei, return_pos=False, silence=True)

    # DEBUG
    # qfk(p_index, off, ik_qua, ik_trs, 1)

    return ik_trs, ik_qua
