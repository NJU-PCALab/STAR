import csv
csv.field_size_limit(8000000)
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from . import video_transforms
from .utils import center_crop_arr
# import video_transforms
# from utils import center_crop_arr

import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import ast


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


#  open-sora-plan+magictime dataset
class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        video_samples = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            #csv_list = list(reader)
            print('csv read end')
            parts_list = os.listdir(root)
            for idx, v_s in enumerate(reader):
                if idx > 0:  # no csv head
                    vid_name = v_s[0]
                    vid_captions_str = v_s[3]
                    vid_captions=ast.literal_eval(vid_captions_str)
                    #vid_captions = vid_captions_str.split('\'')[1::2]
                    for part in parts_list:
                        vids_path = os.path.join(root, part, vid_name)
                        if os.path.isdir(vids_path):
                            for ic, cap in enumerate(vid_captions):
                                vid_path = os.path.join(root, part, vid_name, vid_name+"_"+str(ic)+".mp4")
                                if os.path.exists(vid_path):
                                    video_samples.append([vid_path, cap])
                            break
                if idx % 1000 == 0:
                    print('read video', idx)

        self.samples = video_samples  #
        print('video num:', len(self.samples))

        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0
            
            loop_index = index
            while(total_frames < self.num_frames or is_exit == False):
                #print("total_frames:", total_frames, "<", self.num_frames, ", or", path, "does not exit!!!")
                loop_index += 1
                if loop_index >= len(self.samples):
                    loop_index = 0
                sample = self.samples[loop_index]
                path = sample[0]
                text = sample[1]

                is_exit = os.path.exists(path)
                if is_exit:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    total_frames = len(vframes)
                else:
                    total_frames = 0
            #  video exits and total_frames >= self.num_frames
            
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            
            #print("total_frames:", total_frames, "frame_indice:", frame_indice, "sample:", sample)
            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_path = '/mnt/bn/videodataset-uswest/VDiT/dataset/panda50m/panda70m_training_full.csv'
    root='/mnt/bn/videodataset-uswest/panda70m'
    dataset = DatasetFromCSV(
        data_path,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
        root=root,
    )
    sampler = DistributedSampler(
    dataset,
    num_replicas=1,
    rank=0,
    shuffle=True,
    seed=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    for video_data in loader:
        print(video_data)