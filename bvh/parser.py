import re
from collections import OrderedDict  # NOTE: since python >= 3.6, OrderedDict == dict
import pickle
import gzip


class JointOffset:
    """
        joint name
        joint parent name
        joint child(ren)
        offset
        channel
        order
    """
    def __init__(self, name: str = "Default", parent_name: str = "None",
                 children_names: list = None, offset: list = None, channel: int = 3, order: str = 'ZYX'):
        self.name = name
        self.parent_name = parent_name
        self.children_names = [] if children_names is None else children_names.copy()
        self.offset = [0.0, 0.0, 0.0] if offset is None else offset.copy()
        self.channel = channel
        self.order = order


class JointMotion:
    """
        joint name
        motion data (list of list)  // len(data) == #frames, len(data[0]) == channel
    """
    def __init__(self, name: str = "Default", motion: list = None):
        self.name = name
        self.data = [] if motion is None else motion.copy()


class BVHInfo:
    """
    only load the meta data of a *.bvh file
    """
    def __init__(self, bvh_filepath=""):
        self.frames = 0
        self.frame_time = 0.0
        if bvh_filepath:
            self.from_file(bvh_filepath)

    def from_file(self, bvh_filepath: str) -> None:
        self.filepath = bvh_filepath
        if bvh_filepath.endswith(".bvh"):
            self._from_bvh(bvh_filepath)
        elif bvh_filepath.endswith(".bpk"):
            self._from_bpk(bvh_filepath)
        else:
            raise NotImplementedError(f"Expecting a bvh file or bpk file, but got {bvh_filepath}")

    def _from_bpk(self, bvh_filepath) -> None:
        with gzip.open(bvh_filepath, "rb", compresslevel=1) as f:
            bpk = pickle.load(f)
        self.frames = bpk['f']
        self.frame_time = bpk['t']

    def _from_bvh(self, bvh_filepath):
        info_counter = 0
        info_max = 2
        with open(bvh_filepath) as f:
            for file_line in f:
                file_line = file_line.strip()
                if file_line == '':
                    continue

                split_words = re.split(r'\s+', file_line)
                split_words[0] = split_words[0].upper()

                if split_words[0] == 'FRAMES:':
                    self.frames = int(split_words[1])
                    info_counter += 1
                elif split_words[0] == 'FRAME' and split_words[1].upper() == 'TIME:':
                    self.frame_time = float(split_words[2])
                    info_counter += 1
                if info_counter >= info_max:
                    break

    @property
    def fps(self):
        if self.frame_time == 0: return -1
        else: return int(1.0 / self.frame_time + 0.5)

    # TODO
    # names, offsets, p_index, and more ...


class BVH:
    """
    a BVH object is used to from_file a *.bvh file, parse it, and store the results
    example:
        bvh = BVH()
        bvh.from_file('./data/assets/test.bvh')
        bvh.offset_data is a dictionary that stores skeleton data:  (names sorted in dfs order)
            {
                'Root': a `JointOffset` object
                'Neck': a `JointOffset` object
                'Head': a `JointOffset` object
                ...
            }
        bvh.motion_data is a dictionary that stores the motion data:  (names sorted in dfs order)
            {
                'Root': a `JointMotion` object
                'Neck': a `JointMotion` object
                'Head': a `JointMotion` object
                ...
            }
        bvh.root_name
        bvh.frames
        bvh.frame_time
        bvh.to_file('./temp/test.bvh')
    """
    def __init__(self, bvh_filepath=""):
        self.offset_data = OrderedDict()
        self.motion_data = OrderedDict()
        self.frames = 0
        self.frame_time = 0.0
        self.root_name = ""
        self.filepath = bvh_filepath
        if bvh_filepath:
            self.from_file(bvh_filepath)

    @property
    def rotation_order(self):
        """
        NOTE: this function assumes that the rotation order of all joints are the same
        :return: the rotation order of the root joint
        """
        if self.root_name not in self.offset_data:
            return None
        channel = self.offset_data[self.root_name].channel
        order = self.offset_data[self.root_name].order
        return order[3:] if channel == 6 else order[:3]

    def dfs(self, name=None, depth=0):
        """
        deep first search and return names
        :param name: `None` for root name
        :param depth: depth of that node
        :return: yield name: str, depth: int
        """
        if name is None: name = self.root_name
        yield name, depth
        for ch_name in self.offset_data[name].children_names:
            for nm, dp in self.dfs(ch_name, depth+1):
                yield nm, dp

    def dfs_index(self):
        """
        joint index [0, 1, 2, 3, 4, ...] (same order as the .bvh file)
        :return: a list where list[i] == joint_index(`dfs_ordered_name`)
        """
        names = list(self.offset_data.keys())
        return [names.index(e) for e, _ in self.dfs()]

    def dfs_parent(self):
        """
        parent index [-1, 0, 1, 2, ...] where -1 stands for root node's parent (no parent)
        :return: a list where list[i] == parent_index(i)
        """
        names = [''] + list(self.offset_data.keys())
        return [names.index(self.offset_data[e].parent_name) - 1 for e, _ in self.dfs()]

    def get_indices_of_joints(self, joint_names: list):
        """
        e.g.
            Hand:1 --- Index:2
                   |
                   +-- Thumb:3
            given [Hand, Thumb] returns [1, 3]

        NOTE: since the order of dfs is certain, the index of each joint is fixed.

        :param joint_names: a list of joint names
        :return: a list of int
        """
        names = [e[0] for e in self.dfs()]
        indices = [names.index(nm) for nm in joint_names]
        return indices

    def to_file(self, bvh_filepath: str, pickle_compression=False) -> None:
        if pickle_compression:
            if bvh_filepath.endswith(".bvh"):
                bvh_filepath = bvh_filepath[:-4] + '.bpk'
            else:
                assert bvh_filepath.endswith(".bpk"), "extension should be .bpk (bvh pickle)"
            self._to_bpk(bvh_filepath)
        else:
            assert bvh_filepath.endswith(".bvh"), "extension should be .bvh"
            self._to_bvh(bvh_filepath)

    def _to_bvh(self, bvh_filepath) -> None:
        with open(bvh_filepath, 'w') as f:
            # --------------------------- OFFSET PART --------------------------- #
            f.write('HIERARCHY\n')
            last_depth = -1
            for name, depth in self.dfs():
                while last_depth >= depth:
                    f.write(' ' * (last_depth * 4))
                    f.write('}\n')
                    last_depth -= 1

                if depth != 0: f.write(' '*(depth * 4))
                f.write('ROOT ' if depth == 0 else 'JOINT ')
                f.write(f'{name}\n')

                if depth != 0: f.write(' '*(depth * 4))
                f.write('{\n')

                offset = self.offset_data[name]

                f.write(' '*(depth * 4 + 4))
                f.write(f'OFFSET {offset.offset[0]:.6f} {offset.offset[1]:.6f} {offset.offset[2]:.6f}\n')

                f.write(' '*(depth * 4 + 4))
                f.write(f'CHANNELS {offset.channel} ')
                if offset.channel == 6:
                    f.write(f'Xposition Yposition Zposition ')
                ch = max(offset.channel-3, 0)  # XYZ, ZYX => 0 or XYZ|XYZ, XYZ|ZYX => 3
                f.write(f'{offset.order[0+ch]}rotation {offset.order[1+ch]}rotation {offset.order[2+ch]}rotation\n')

                if len(offset.children_names) == 0:
                    if depth != 0: f.write(' ' * (depth * 4 + 4))
                    f.write('End Site\n')
                    if depth != 0: f.write(' ' * (depth * 4 + 4))
                    f.write('{\n')
                    if depth != 0: f.write(' ' * (depth * 4 + 8))
                    f.write('OFFSET 0.000000 0.000000 0.000000\n')
                    if depth != 0: f.write(' ' * (depth * 4 + 4))
                    f.write('}\n')

                last_depth = depth

            # pad more '}' ...
            while last_depth >= 0:
                f.write(' ' * (last_depth * 4))
                f.write('}\n')
                last_depth -= 1

            # --------------------------- MOTION PART --------------------------- #
            f.write('MOTION\n')
            f.write(f'Frames: {self.frames}\n')
            f.write(f'Frame Time: {self.frame_time:.6f}\n')
            for i in range(self.frames):
                for name, _ in self.dfs():  # this would be the same order as in `OFFSET PART`
                    channel = self.offset_data[name].channel
                    motion = self.motion_data[name]
                    for j in range(channel):
                        f.write(f'{motion.data[i][j]:.6f} ')
                f.write('\n')
    
    def _to_bpk(self, bvh_filepath) -> None:
        bpk = {}
        bpk['o'] = self.offset_data
        bpk['m'] = self.motion_data
        bpk['f'] = self.frames
        bpk['t'] = self.frame_time
        bpk['r'] = self.root_name
        with gzip.open(bvh_filepath, "wb", compresslevel=1) as f:
            pickle.dump(bpk, f)
        
    def from_file(self, bvh_filepath: str) -> None:
        self.filepath = bvh_filepath
        if bvh_filepath.endswith(".bvh"):
            self._from_bvh(bvh_filepath)
        elif bvh_filepath.endswith(".bpk"):
            self._from_bpk(bvh_filepath)
        else:
            raise NotImplementedError(f"Expecting a bvh file or bpk file, but got {bvh_filepath}")

    def _from_bpk(self, bvh_filepath) -> None:
        with gzip.open(bvh_filepath, "rb", compresslevel=1) as f:
            bpk = pickle.load(f)
        self.offset_data = bpk['o']
        self.motion_data = bpk['m']
        self.frames = bpk['f']
        self.frame_time = bpk['t']
        self.root_name = bpk['r']

    def _from_bvh(self, bvh_filepath) -> None:
        """
        from_file data from bvh file
        :param bvh_filepath: bvh filepath
        :return:
        """
        # clear data
        self.offset_data = OrderedDict()
        self.motion_data = OrderedDict()
        self.frames = 0
        self.frame_time = 0.0
        self.root_name = ""

        name_order = []  # record name order in the bvh file
        flag_motion = False  # set to True if 'MOTION' appears
        with open(bvh_filepath) as f:
            stack = []  # for dfs
            for file_line in f:
                file_line = file_line.strip()
                if file_line == '':
                    continue

                split_words = re.split(r'\s+', file_line)
                split_words[0] = split_words[0].upper()

                if split_words[0] == 'ROOT':
                    joint = JointOffset(split_words[1], "", [], [], 0, "")
                    self.offset_data[joint.name] = joint
                    stack.append(joint)
                    name_order.append(joint.name)

                    self.root_name = split_words[1]  # set to root

                elif split_words[0] == 'JOINT':
                    joint = JointOffset(split_words[1], "", [], [], 0, "")
                    self.offset_data[joint.name] = joint
                    stack.append(joint)
                    name_order.append(joint.name)

                    stack[-2].children_names.append(split_words[1])  # add child name
                    joint.parent_name = stack[-2].name  # set parent name

                elif split_words[0] == 'END' and split_words[1].upper() == 'SITE':
                    # stk[-1].append( ... )
                    stack.append(None)

                elif split_words[0] == 'OFFSET':
                    if stack[-1] is not None:  # not an end effector
                        stack[-1].offset = [float(e) for e in split_words[1:]]

                elif split_words[0] == 'CHANNELS':
                    stack[-1].channel = int(split_words[1])
                    stack[-1].order = ''.join([e[0] for e in split_words[2:]])

                elif split_words[0] == '}':
                    stack.pop(-1)

                elif split_words[0] == 'MOTION':
                    flag_motion = True
                    # create an empty dictionary to store motion data
                    for name in name_order:
                        self.motion_data[name] = JointMotion(name, [])

                elif split_words[0] == 'FRAMES:':
                    self.frames = int(split_words[1])

                elif split_words[0] == 'FRAME' and split_words[1].upper() == 'TIME:':
                    self.frame_time = float(split_words[2])

                elif flag_motion:
                    i = 0
                    data_of_frame = [float(e) for e in split_words]
                    for name in name_order:
                        channel = self.offset_data[name].channel
                        self.motion_data[name].data.append(data_of_frame[i:i + channel])
                        i += channel

    def print_topology_info(self):
        print("BVH topology --- >>>")
        for i, (name, depth) in enumerate(self.dfs()):
            print('-'*depth, name, '----', i)
        print("<<< --- BVH topology")
    
    @property
    def p_index(self):
        return self.dfs_parent()

    @property
    def names(self):
        return [e for e, _ in self.dfs()]

    @property
    def fps(self):
        if self.frame_time == 0: return -1
        else: return int(1.0 / self.frame_time + 0.5)
