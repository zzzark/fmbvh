# @author: zrk
# @desc:   utils for motion upsampling, calc pivot rotation along y-axis , foot contact, etc.

import torch
import torch.nn.functional as F
from . import rotations


def sample_frames(motion: torch.Tensor, scale_factor=None, target_frame=None, sampler='nearest'):
    """
    upsample a given rotation to a certain frame_time, scale_factor, ... ...
    :param motion: JxCxF   (J: num_joints; C: channel, 3(euler, position) or 4(quaternion); F: num_frames)
    :param scale_factor: source_frame_time / target_frame_time
    :param target_frame: how many frames are needed
                         NOTE: if scale_factor is not none, another interpolate will be applied
                               to obtain `target_frame` frames
    :param sampler: 'nearest', 'linear', etc.
    :return:
    """
    assert len(motion.shape) == 3, 'sample_frames: input should be [J, C, F]'

    if scale_factor is not None and abs(scale_factor - 1.0) > 1e-3:
        # noinspection PyArgumentList
        motion = F.interpolate(motion, size=None, recompute_scale_factor=False, scale_factor=scale_factor, mode=sampler)

    if target_frame is not None and motion.shape[-1] != target_frame:  # frames not aligned yet
        # noinspection PyArgumentList
        motion = F.interpolate(motion, size=target_frame, recompute_scale_factor=False, scale_factor=None, mode=sampler)

    return motion


def align_root_rot(pos: torch.Tensor, root_rot, hip: tuple, sho: tuple, to_axis='Z', up_axis='Y', 
                   return_pos=False, set_pose_origin=True):
    """
    align root rotation to a certain direction (x/y/z-axis)
    :param pos:  [(B), J, 3, F], positions
    :param root_rot: [(B), 4, F], rotations
    :param hip: (L-hip, R-hip)
    :param sho:  (L-shoulder, R-shoulder)
    :param to_axis: axis to align (to face towards)
    :param up_axis: up axis
    :param return_pos: return laigned pose or not
    :param set_pose_origin: move root to origin or not
    :return: [(B), 4, F] for new root rotation (towards `to_axis` axis), [(B), 1, F] for rotation radius along `up_axis`
    """
    assert isinstance(hip, tuple) and isinstance(sho, tuple)
    assert hip[0] is not None and hip[1] is not None and sho[0] is not None and sho[1] is not None

    h = pos[..., hip[0], :, :] - pos[..., hip[1], :, :]  # L<-R
    s = pos[..., sho[0], :, :] - pos[..., sho[1], :, :]  # L<-R
    ve = (h + s) * 0.5
    up = torch.zeros_like(ve, device=pos.device)
    up[..., 'XYZ'.index(up_axis), :] = 1.0

    forward = torch.cross(ve, up, dim=-2)
    forward = torch.nn.functional.normalize(forward, p=2.0, dim=-2)  # [1, 3, F]
    forward[..., 'XYZ'.index(up_axis), :] = 0.0  # proj to XZ plane
    forward = torch.nn.functional.normalize(forward, p=2.0, dim=-2)

    to_dir = torch.zeros_like(forward, device=pos.device)
    to_dir[..., 'XYZ'.index(to_axis), :] = 1.0

    to_dir_qua = rotations.quaternion_from_two_vectors(forward, to_dir)  # [4, F]
    w = to_dir_qua[..., 0, :]
    root_eul_y = torch.arccos(w) * 2  # theta
    # TODO: smooth root_eul_y and recompute to_dir_qua

    new_root = rotations.mul_two_quaternions(to_dir_qua, root_rot)  # [(B), 4, F]
    new_root = rotations.rectify_w_of_quaternion(new_root)  # [(B), 4, F]

    if return_pos:
        new_pos = pos - pos[..., [0], :, :]
        new_pos = rotations.rotate_vector_with_quaternion(new_pos, to_dir_qua[..., None, :, :])
        if not set_pose_origin:
            new_pos = new_pos + pos[..., [0], :, :]
        return new_root, root_eul_y, new_pos
    else:
        return new_root, root_eul_y


def align_root_rot_old(pos: torch.Tensor, root_rot, hip: tuple, sho: tuple, to_axis='Z', up_axis='Y'):
    """
    align root rotation to a certain direction (x/y/z-axis)
    :param pos:  [(B), J, 3, F], positions
    :param root_rot: [(B), 4, F], rotations
    :param to_axis: axis to align (to face towards)
    :param up_axis: up axis
    :param hip: (L-hip, R-hip)
    :param sho:  (L-shoulder, R-shoulder)
    :return: [(B), 4, F] for new root rotation (towards `to_axis` axis), [(B), 1, F] for rotation radius along `up_axis`
    """
    batch = True if len(pos.shape) == 4 else False
    if not batch:
        pos = pos[None, ...]
        root_rot = root_rot[None, ...]

    if up_axis != 'Y':
        raise NotImplementedError

    h = pos[:, hip[0], ...] - pos[:, hip[1], ...]  # L<-R
    s = pos[:, sho[0], ...] - pos[:, sho[1], ...]  # L<-R
    ve = (h + s) * 0.5
    up = torch.zeros_like(ve, device=pos.device)  # [B, 3, F]
    up[:, 'XYZ'.index(up_axis), :] = 1.0

    forward = torch.cross(ve, up, dim=1)
    forward = torch.nn.functional.normalize(forward, p=2.0, dim=1)

    to_dir = torch.zeros_like(forward, device=pos.device)
    to_dir[..., 'XYZ'.index(to_axis), :] = 1.0

    to_dir_qua = rotations.quaternion_from_two_vectors(forward, to_dir)
    new_root = rotations.mul_two_quaternions(to_dir_qua, root_rot)
    neg_w = (new_root[:, [0], :] < 0).expand(-1, 4, -1)
    new_root[neg_w] = -new_root[neg_w]
    root_eul_y = rotations.quaternion_to_euler(to_dir_qua[:, None, :, :], 'XZY', intrinsic=False)[:, 0, [2], :]

    if not batch:
        new_root = new_root[0]
        root_eul_y = root_eul_y[0]

    return new_root, root_eul_y


def calc_joint_offset(trs: torch.Tensor):
    """
    get offset from position (root translation)

    offset of frame i     ===> O_{i} = F_{i} - F_{i-1}
    offset of first frame ===> O_{0} = 0.5 * (O_{1} + O_{2})
    :param trs: root translation (1x3xF)
    :return: 1x3xF
    """
    _, _, f = trs.shape
    off = torch.empty_like(trs, device=trs.device)
    off[:, :, 1:f] = trs[:, :, 1:f] - trs[:, :, 0:f-1]
    off[:, :, 0] = 0.5 * (off[:, :, 1] + off[:, :, 2])   # approximate the first frame
    return off


def sum_joint_offset(off: torch.Tensor, world=None):
    """
    sum up all the offset vectors to obtain position

    NOTE:
        world = frame_0_absolute_position - frame_0_offset_vector
    that means:
        world = frame_{-1}_absolute_position   (-1 <==> last frame)
    since:
        frame_0_absolution_position = frame_{-1}_absolution_position + frame_0_offset_vector

    :param off:    root translation (1x3xF)
    :param world:  world position (is a tensor: 1x3x1, not a tensor: len==3)
    :return: 1x3xF
    """
    trs = torch.cumsum(off, dim=2)
    if world is not None:
        world = torch.tensor(world, device=trs.device)
        world = world.view(1, -1, 1)
    if isinstance(world, torch.Tensor):
        trs += world
    return trs


def pad_motion(motion: torch.Tensor, l_pad=0, r_pad=0, mode='reflect'):
    return F.pad(motion, (l_pad, r_pad), mode, value=0)


def get_motion_masked(motion: torch.Tensor, mask: list) -> torch.Tensor:
    """
    get a subset of motion
    e.g.
        Hand:1 --- Index:2
               |
               +-- Thumb:3

        mask = [1, 2] will cut off the `Thumb` node and returns [Hand, Index]

    :param motion: JxNxF or BxJxNxF
    :param mask: a list of `int`
    :return: JxMxF or BxJxMxF,  where M = N - len(mask)
    """
    return motion[..., mask, :, :]


def get_feet_contacts(pos: torch.Tensor, ee_ids: list,
                      ref_height: float, 
                      criteria='pos',
                      vel_thres=0.005,
                      pos_thres=0.030,
                      up_axis=1,
                      kernel_size=7):
    """
    get pseudo-fc labels

    :param pos: [..., J, 3, T]
    :param ee_ids: end-effectors' indices
    :param ref_height: body height
    :param criteria: 'vel' or 'pos'
    :param vel_thres: threshold (divided by body height)
    :param pos_thres: threshold (divided by body height)
    :param up_axis: up axis, x-0, y-1, z-2
    :param kernel_size: kernel size for median filter (recommended: 11-120FPS; 7-60FPS; 5-30FPS)

    :return: fc [..., E, T]  (int, 1 for contacted and 0 for not)
    """
    ee_pos = pos[..., ee_ids, :, :]  # [..., E, 3, T]
    Y = up_axis
    
    if criteria == 'vel':
        ee_velo = ee_pos[..., 1:] - ee_pos[..., :-1]
        ee_velo = ee_velo / ref_height
        ee_velo_norm = torch.norm(ee_velo, dim=-2)  # [..., E, T]
        contact = ee_velo_norm < vel_thres
        contact = contact.float()
        padding = torch.zeros_like(contact[..., :1])
        contact = torch.cat([padding, contact], dim=-1)
    elif criteria == 'pos':
        contact = (ee_pos[..., Y, :] < pos_thres * ref_height)
        contact = contact.float()
    else:
        raise NotImplementedError
    
    def __median_filter(tensor, ks, dim):
        lp = (kernel_size - 1) // 2
        rp = (kernel_size - 1) - lp
        padded_tensor = F.pad(tensor, (lp, rp), mode='reflect')
        unfolded_tensor = padded_tensor.unfold(dim, ks, 1)
        median = unfolded_tensor.median(dim=dim).values
        return median
    
    # filter salt and pepper noise
    if kernel_size >= 3:
        contact = __median_filter(contact, kernel_size, -1)

    return contact.int()


@torch.no_grad()
def get_feet_grounding_shift(fp, fc, up_axis=1, iter_=2, kernel=7) -> float:
    """
    get how far the feet drift away from floor (return the up-axis value)
    :param fp: [E, 3, T], original foot position
    :param fc: [E, T], soft foot contact
    :return [T]
    """
    Y = up_axis
    T = fc.shape[-1]
    K = kernel if kernel%2 else kernel-1
    assert K >= 3
    K2 = (K-1)//2
    ikp = torch.zeros_like(fc, dtype=fp.dtype)  # ik position, [E, T]
    mask = (fc != 0)
    ikp[mask] = fp[:, Y, :][mask]
    org_ikp = ikp.clone()

    for _ in range(iter_):
        tmp = torch.zeros_like(ikp)
        for t in range(0, T):
            for k in range(-K2, K2+1):
                i = t+k
                if i < 0:  i = -i
                if i >= T: i = (T+T-1) - i
                tmp[:, t] += ikp[:, i]
            tmp[:, t] *= (1/K)  # mean filter
        ikp = tmp
        # ikp[~mask] = tmp[~mask]

    # from ..visualization.nplots import visualize_arrays
    # visualize_arrays([org_ikp, ikp])

    ikp = ikp.mean(dim=0)  # [E, T] -> [T]
    return ikp


def __lerp(a, l, r):
    return (1 - a) * l + a * r


def __alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


@torch.no_grad()
def get_feet_contact_points(fp: torch.Tensor, fc: torch.Tensor,
                            force_on_floor=True, interp_length=10, up_axis=1) -> torch.Tensor:
    """
    given original feet positions and contact labels, return smoothed feet positions for IK

    :param fp: [E, 3, T], original foot position
    :param fc: [E, T], soft foot contact
    :param force_on_floor: force the feet on floor (y=0)
    :param interp_length:
    :param up_axis: 
    :return: [E, 3, T], desired foot position on ground
    """
    assert len(fp.shape) == 3, "batch not supported"
    assert len(fc.shape) == 2, "batch not supported"
    Y = up_axis

    res = []
    for pos, ctc in zip(fp.clone(), fc):
        T = ctc.shape[-1]

        s = 0
        while s < T:
            while s < T and ctc[s] == 0: s += 1
            if s >= T: break
            t = s

            # compute an averaged position within the "contacting window"
            avg = pos[:, t].clone()
            while t + 1 < T and ctc[t + 1] == 1:
                t += 1
                avg += pos[:, t].clone()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[Y] = 0.0

            # and assign this averaged value to the window
            for j in range(s, t + 1):
                pos[:, j] = avg.clone()

            s = t + 1
        res.append(pos)

        # then make the transition smoother
        for s in range(T):
            if ctc[s] == 1: continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if ctc[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if ctc[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue

            if consl and consr:
                litp = __lerp(__alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                              pos[:, s], pos[:, l])
                ritp = __lerp(__alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                              pos[:, s], pos[:, r])
                itp = __lerp(__alpha(1.0 * (s - l + 1) / (r - l + 1)),
                             ritp, litp)
                pos[:, s] = itp.clone()
                continue
            if consl:
                litp = __lerp(__alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                              pos[:, s], pos[:, l])
                pos[:, s] = litp.clone()
                continue
            if consr:
                ritp = __lerp(__alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                              pos[:, s], pos[:, r])
                pos[:, s] = ritp.clone()

    return torch.stack(res, dim=0)
