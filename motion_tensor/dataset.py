from .dataset_core import *
from torch.utils.data import Dataset, DataLoader
from ..bvh import parser


class SimpleDatasetNoStaticNoClass(Dataset):
    class __SimpleExtractor(BVHDataExtractor):
        def __init__(self, fps, proc_fn):
            """
            :param proc_fn: proc_fn(bvh_obj, sampler_fn) -> feature: torch.Tensor[..., F]
            """
            super().__init__(desired_frame_time=1.0/fps)
            self.fn = proc_fn

        def extract(self, bvh_obj: parser.BVH) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
            __samp = lambda e: self.scale(e, bvh_obj.frame_time)
            return None, self.fn(bvh_obj, __samp)

    class __SimpleProcessor(MoClipProcessor): pass

    class __SimpleCollector(MoStatisticCollector):
        @staticmethod
        def __get_stat(feature: List[tuple]) -> Any:
            feature =  [e[1] for e in feature]
            m = torch.mean(torch.concat(feature, dim=-1), dim=(-1))
            v = torch.var(torch.concat(feature, dim=-1), dim=(-1))
            s = (v + 1e-6).sqrt()
            return m, v, s

        def get_stat(self, class_id: int, feature: List[tuple]) -> Any:
            return self.__get_stat(feature)
        
        def get_stat_all(self, feature: List[Tuple]) -> Any:
            return self.__get_stat(feature)

    def __init__(self, 
                 in_folder, out_folder,
                 fps, proc_fn, 
                 window, window_step, skip=0,
                 divider_override=None,  # override for handling flip
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.extractor = self.__SimpleExtractor(fps, proc_fn)
        self.stat_dic = gather_statistic(in_folder, out_folder, self.extractor, self.__SimpleProcessor(), self.__SimpleCollector())

        # NOTE: can be normalized when saving
        if divider_override is not None:
            assert isinstance(divider_override, MotionDataDivider)
        else:
            divider_override = MotionDataDivider(window, window_step, skip=skip)
        make_mo_clip_dataset(in_folder, out_folder, self.extractor, divider_override)
        self.dataset = load_mo_clip_dataset(out_folder, self.__SimpleProcessor())

        from datetime import timedelta
        print("Simple Dataset Summary === >>>")
        print(f"    frames of original data: {self.stat_dic['frames']}")
        sec = int(self.stat_dic['frames'] / fps)
        print(f"    - total duration: {timedelta(seconds=sec)}")

        if 'per_frames' in self.stat_dic:
            ls = [str(timedelta(seconds=int(e / fps))) for e in self.stat_dic['per_frames']]
            print(f"    - per-class duration: \n        {ls}")

        # print(f"    frames of processed data: {len(self) * window}")
        # sec = int(len(self) * window / fps)
        # print(f"    - duration: {timedelta(seconds=sec)}")
        print("<<< === Simple Dataset Summary")

    
    def process(self, obj: parser.BVH):  # bvh -> feat
        return self.extractor.extract(obj)[1]  # dynamic-part only

    def revert(self):  # feat -> bvh
        raise NotImplementedError

    @property
    def mean(self): return self.stat_dic["all"][0]

    @property
    def var(self): return self.stat_dic["all"][1]
    
    @property
    def std(self): return self.stat_dic["all"][2]

    @staticmethod
    def _broadcast(dst, src):
        if len(dst.shape) - len(src.shape) == 1:  # ... frame
            src = src[..., None]
        elif len(dst.shape) - len(src.shape) == 2:  # batch ... frame
            src = src[None, ..., None]
        elif len(dst.shape) - len(src.shape) == 2:  # batch ... frame
            pass
        else:
            raise NotImplementedError
        return src

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        m = self._broadcast(x, self.mean)
        v = self._broadcast(x, self.std)
        return (x - m) / v
    
    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        m = self._broadcast(x, self.mean)
        v = self._broadcast(x, self.std)
        return x * v + m

    def get_loader(self, batch_size, shuffle, num_workers, *args, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size, shuffle, num_workers=num_workers, *args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sta, dyn, cls = self.dataset[item]
        return dyn
