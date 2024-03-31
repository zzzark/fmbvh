"""
gather all the *.bvh files and load them into memory as BVH objects
"""

import os
from typing import Tuple
from .. import bvh
from ..bvh import parser
import glob


class BVHSubFolder:
    def __init__(self, sub_folder_path: str=""):
        self.file_list = []
        self.bvh_cache = {}
        self.folder_path = sub_folder_path

        if sub_folder_path != "":
            self.create_dataset(sub_folder_path)

    def create_dataset(self, bvh_file_folder: str) -> None:
        self.file_list = []
        import os.path as path
        for file in glob.iglob(path.join(bvh_file_folder, "*.bvh"), recursive=False):
            self.file_list.append(file)
        for file in glob.iglob(path.join(bvh_file_folder, "*.bpk"), recursive=False):
            self.file_list.append(file)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item) -> parser.BVH:
        filename = self.file_list[item]  # may raise an IndexError
        if filename not in self.bvh_cache:
            bvh_obj = parser.BVH(self.file_list[item])
            self.bvh_cache[filename] = bvh_obj
        return self.bvh_cache[filename]


# Load BVH files from multiple folders with index, sorted by filename
class BVHFolder:
    def __init__(self, bvh_file_folder=""):
        self.dataset_list = []

        if bvh_file_folder != "":
            self.create_dataset(bvh_file_folder)

    def __len__(self):
        return sum([len(e) for e in self.dataset_list])

    def __getitem__(self, item: int) -> Tuple[int, parser.BVH]:
        cls = 0
        for ds in self.dataset_list:
            if item >= len(ds):
                item -= len(ds)
                cls += 1
            else:
                obj = ds[item]
                return cls, obj
        raise IndexError

    def __iter__(self):
        for cls, sub in enumerate(self.dataset_list):
            for obj in sub:
                yield cls, obj

    def create_dataset(self, bvh_file_folder: str, name_ascending=True) -> None:
        for file in sorted(os.listdir(bvh_file_folder), reverse=not name_ascending):
            subdir = os.path.join(bvh_file_folder, file)
            if os.path.isdir(subdir):
                inst_dataset = BVHSubFolder(subdir)
                self.dataset_list.append(inst_dataset)
        if len(self.dataset_list) == 0:  # if no subfolder then load the top folder as subfolder
            self.dataset_list.append(BVHSubFolder(bvh_file_folder))
