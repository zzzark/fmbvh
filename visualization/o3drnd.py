import open3d as o3d
from typing import Iterator
import time
import torch
from ..motion_tensor.rotations import quaternion_to_matrix as q2m
from ..motion_tensor.kinematics import forward_kinematics as fk


class _CallbackContainer:
    def __init__(self, fun, renderer):
        self.fun = fun
        self.rnd = renderer

    def __call__(self, *args, **kwargs):
        self.fun(self.rnd)


class O3DRenderer:
    def __init__(self, left=400, top=300, width=1600, height=1200):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=left, top=top)
        self.geometries_set = {}

    def create_window(self, left=400, top=300, width=1600, height=1200):
        self.vis.destroy_window()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=left, top=top)
        self.geometries_set = {}

    def show_window(self) -> None:
        """
        run until window closed
        :return: None
        """
        self.vis.run()
        self.vis.destroy_window()

    def set_keyboard_callback(self, key: str, fun: callable) -> None:
        """
        callback function
        :param key: 'a' ~ 'z' or 'A' ~ 'Z'
        :param fun: fun = def your_callback_function(obj: O3DRenderer)
        :return: None
        """
        self.vis.register_key_callback(ord(key.upper()[0]), _CallbackContainer(fun, self))

    def set_animation_callback(self, fun: callable) -> None:
        """
        set animation callback function
        :param fun: fun = def your_callback_function(obj: O3DRenderer)
        :return: None
        """
        self.vis.register_animation_callback(_CallbackContainer(fun, self))

    def is_name_exists(self, name):
        return name in self.geometries_set

    def add_lines(self, name, points, lines, colors=None) -> None:
        """
        add new lines with a unique name
        :param name: geometry name
        :param points: Nx3 list for N points
        :param lines:  Kx2 list for K lines, index starts from 0
        :param colors: Kx3 list for K lines, range from 0 ~ 1
        :return: None
        """
        if name in self.geometries_set:
            raise KeyError(f'Key "{name}" already exists!')

        if colors is None:
            colors = [[0, 0, 0] for _ in range(len(lines))]

        # range check
        assert len(lines) == len(colors), f"size not match for lines: {len(lines)} and colors: {len(colors)}"
        N = len(points)
        for p in lines:
            for e in p:
                assert e < N, f"index out of range: {e} > {N - 1}"

        point_set = o3d.geometry.PointCloud()
        point_set.points = o3d.utility.Vector3dVector(points)
        point_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(N)])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)
        self.vis.add_geometry(point_set)

        geo_dic = {
            'lines': line_set,
            'points': point_set,
            'N': len(points),
            'K': len(lines)}
        self.geometries_set[name] = geo_dic

    def set_lines(self, name, points=None, lines=None, colors=None) -> None:
        """
        set lines with the given name
        :param name: geometry name
        :param points: points
        :param lines:
        :param colors:
        :return:
        """
        geo_dic = self.geometries_set[name]
        line_set = geo_dic['lines']
        point_set = geo_dic['points']

        if points is not None:
            line_set.points = o3d.utility.Vector3dVector(points)
            point_set.points = o3d.utility.Vector3dVector(points)
            geo_dic['N'] = len(points)

        if lines is not None:
            N = geo_dic['N']
            for p in lines:
                for e in p:
                    assert e < N, f"index out of range: {e} > {N-1}"

            line_set.lines = o3d.utility.Vector2iVector(lines)
            geo_dic['K'] = len(lines)

        if colors is not None:
            K = geo_dic['K']
            assert K == len(colors), f"size not match for lines: {K} and colors: {len(colors)}"
            line_set.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(line_set)
        self.vis.update_geometry(point_set)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.geometries_set[name] = geo_dic


def demo_1():
    def fun_a(rnd: O3DRenderer):
        rnd.set_lines('lines', [[1, 2, 3],
                                [1, 4, 3],
                                [2, 1, 3],
                                [1, 1, 0]],
                      [[0, 1], [1, 2], [2, 3]])

    def fun_b(rnd: O3DRenderer):
        # ERROR: out of range: 4
        rnd.set_lines('lines', [[1, 2, 3],
                                [1, 4, 3],
                                [2, 1, 3],
                                [1, 1, 0]],
                      [[1, 2], [2, 3], [3, 4]])

    def fun_ani(rnd: O3DRenderer):
        # ERROR: press `A` can cause a size not match error
        import time
        time.sleep(1 / 60.0)  # 60FPS
        colors = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0.5, 0.5, 0]]
        import random
        cl = [random.choice(colors) for _ in range(4)]
        rnd.set_lines('lines', colors=cl)

    renderer = O3DRenderer()
    renderer.add_lines('lines', [[1, 2, 3],
                                 [1, 4, 3],
                                 [2, 1, 3],
                                 [1, 1, 0]],
                      [[0, 1], [1, 2], [2, 3], [3, 0]])
    renderer.set_keyboard_callback('A', fun_a)
    renderer.set_keyboard_callback('B', fun_b)
    renderer.set_animation_callback(fun_ani)
    renderer.show_window()


class MoVisualizer:
    def __init__(self, p_index: list, mo_source: Iterator[list], max_fps=60.0, add_coordinate=True, scale=1.0):
        """
        :param p_index:  a list of parent index
        :param mo_source: a iterator that yields position for next frame, end with `None`
                          yields a [N, 3] list
        :param max_fps:   maximum fps
        """
        self.rnd = O3DRenderer()
        self.p_index = p_index
        self.mo_source = mo_source
        self.max_fps = max_fps

        # add coordinate
        if add_coordinate:
            coord_p = [[-1, 0, 0], [+1, 0, 0], [0, 0, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]]
            coord_l = [[0, 1], [2, 3], [4, 5]]
            coord_p = [[e*scale for e in v] for v in coord_p]
            self.rnd.add_lines('__default__$coord', coord_p, coord_l)

        # add skeleton
        self.rnd.add_lines('motion',
                           [[0, 0, 0] for _ in range(len(p_index))],
                           [[i, p] if p != -1 else [i, i] for i, p in enumerate(p_index)])
        self.rnd.set_animation_callback(self.__update)

    def add_grids(self, m: int, n: int, scale=1.0) -> None:
        """
        add mxn grids (y = 0)
        :param m: m grids along z-axis
        :param n: n grids along x-axis
        :param scale: scale of object(s)
        :return: None
        """
        grids_p = []
        grids_l = []
        for z in range(m+1):
            for x in range(n+1):
                grids_p.append([x - (n / 2), 0, z - (m / 2)])
        for z in range(m+1):
            for x in range(0, n):
                grids_l.append([x+z*(n+1), x+1+z*(n+1)])
            if z == m:
                break
            for x in range(0, n+1):
                grids_l.append([x+z*(n+1), x+(z+1)*(n+1)])

        grids_p = [[e*scale for e in v] for v in grids_p]
        self.rnd.add_lines('__default__$grids', grids_p, grids_l)

    def run(self):
        self.rnd.show_window()

    def __update(self, _):
        st = time.time()
        points = next(self.mo_source, None)
        if points is not None:
            self.rnd.set_lines('motion', points)
        delta = time.time() - st
        rest = (1.0 / self.max_fps) - delta
        if rest > 0:
            time.sleep(rest)


def demo_2():
    from ..bvh import parser
    bvh_obj = parser.BVH('../data/assets/test.bvh')

    import motion_tensor as mot
    import torch

    trs, qua = mot.bvh_casting.get_quaternion_from_bvh(bvh_obj)  # [1, 3, F], [J, 3, F]
    mat = mot.rotations.quaternion_to_matrix(qua[None, ...])  # [B, J, 3, 3, F]
    J, _, F = qua.shape

    offsets = mot.bvh_casting.get_offsets_from_bvh(bvh_obj)[None, ...]  # [B, J, 3, 1]
    offsets = torch.broadcast_to(offsets, (1, J, 3, F))  # [B, J, 3, F]

    fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
                                               trs[None, ...], offsets)    # [B, J, 3, F]

    t_pos = mot.bvh_casting.get_positions_from_bvh(bvh_obj)[..., 0]  # [J, 3]
    max_y = torch.max(t_pos[:, 1])
    min_y = torch.min(t_pos[:, 1])
    height = (max_y - min_y).item()

    def _next():
        f = 0
        while True:
            rpt = trs[:, :, f]  # [1, 3]
            cps = fk_pos[0, :, :, f]  # [J, 3]
            pos_ls = cps + rpt  # [J, 3]
            yield pos_ls.numpy().tolist()
            f = (f + 1) % F

    p_index = bvh_obj.dfs_parent()
    mvz = MoVisualizer(p_index, _next(), scale=height*2, max_fps=60)
    mvz.add_grids(10, 10, height*0.2)
    mvz.run()


def quick_visualize_fk(p_index, off, qua, trans, scale=200.0):
    """
    off: [J, 3, 1]
    qua: [J, 4, F]
    """
    if len(qua.shape) == 4: # remove batch
        off = off[0]
        qua = qua[0]

    if len(qua.shape) == 3:
        assert len(off.shape) == 3
        if qua.shape[0] == len(p_index) and qua.shape[1] == 4:  # [J, 4, T]
            pass
        else:
            qua = qua[0]
            qua = qua.view(qua.shape[0], 4, qua.shape[-1])
    else:
        raise ValueError(f"incorrect shape: {qua.shape}")

    off = off.detach().cpu()
    qua = qua.detach().cpu()
    mat = q2m(qua)
    pos = fk(p_index, mat, None, off, True, False) + trans

    def _next():
        f = 0
        while True:
            yield pos[..., f].tolist()
            f = (f + 1) % pos.shape[-1]

    mvz = MoVisualizer(p_index, _next(), scale=scale)
    mvz.run()


def quick_visualize(p_index, pos, scale=200.0, callback_fn=None):
    """
    pos: [J, 3, F]
    """
    if len(pos.shape) == 4:  # remove batch
        pos = pos[0]

    assert pos.shape[1] == 3
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu()

    def _next():
        f = 0
        while True:
            yield pos[..., f].tolist()
            f = (f + 1) % pos.shape[-1]
            if callback_fn is not None:
                callback_fn(f)

    mvz = MoVisualizer(p_index, _next(), scale=scale)
    mvz.run()

