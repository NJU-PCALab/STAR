from re import T
import cv2
import numpy as np
import random
import torch
import math
# import os
import torch.nn as nn
import sys
sys.path.append('/mnt/bn/videodataset-uswest/VSR/VSR/opensora/datasets/high_order')
from utils_ import filter2D, USMSharp
from utils_blur import circular_lowpass_kernel, random_mixed_kernels
# from utils_resize import random_resizing
from utils_noise import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils_jpeg import DiffJPEG
from torch.nn import functional as F
from einops import rearrange
import av
import io

class ImageCompressor:
    def __init__(self):
        self.params = {
            'codec': ['libx264', 'h264', 'mpeg4'],
            'codec_prob': [1 / 3., 1 / 3., 1 / 3.],
            'bitrate': [1e4, 1e5]
        }

    def _ensure_even_dimensions(self, img):
        # Ensure width and height are even
        h, w = img.shape[:2]
        if h % 2 != 0:
            img = img[:-1, :]
        if w % 2 != 0:
            img = img[:, :-1]
        return img

    def _apply_random_compression(self, imgs):
        # Convert PyTorch tensor to NumPy array
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

        # Ensure width and height are even
        imgs = [self._ensure_even_dimensions(img) for img in imgs]

        codec = random.choices(self.params['codec'], self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = (img * 255).clip(0, 255)  # Convert to [0, 255] range
                img = img.astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(frame.to_rgb().to_ndarray().astype(np.float32) / 255)  # Convert back to [0, 1] range

        # Convert NumPy array back to PyTorch tensor
        outputs = torch.tensor(outputs).permute(0, 3, 1, 2)

        return outputs


class Degradation(nn.Module):
    def __init__(self, scale, gt_size):
        super(Degradation, self).__init__()
        ### initization JPEF class
        self.jpeger = DiffJPEG(differentiable=False)#.cuda()
        self.usm_sharpener = USMSharp()#.cuda()
        # self.queue_size = 180 #opt.get('queue_size', 180)

        ### global settings
        self.scale = scale
        self.gt_size = gt_size

        ### the first degradation hypermeters ###
        # 1. blur
        self.blur_kernel_size = 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]  # blur_x / y_sigma
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        # 2. resize
        self.updown_type = ["up", "down", "keep"]
        self.mode_list = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        # 3. noise
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        # 4. jpeg
        self.jpeg_range = [30, 95]

        ### the second degradation hypermeters ###
        # 1. blur
        self.second_blur_prob = 0.8

        self.blur_kernel_size2 = 21
        self.kernel_range2 = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        # 2. resize
        self.updown_type2 = ["up", "down", "keep"]
        self.mode_list2 = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        # 3. noise
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        # 4. jpeg
        self.jpeg_range2 = [30, 95]

        self.final_sinc_prob = 0.8

        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        # video compression
        self.compressor = ImageCompressor()

    @torch.no_grad()
    def forward_deg(self, gt):
        ori_h, ori_w = gt.size()[2:4]
        gt_usm = self.usm_sharpener(gt)
        gt_usm_copy = gt_usm.clone()

        # generate kernel
        kernel1 = self.generate_first_kernel()
        kernel2 = self.generate_second_kernel()
        sinc_kernel = self.generate_sinc_kernel()

        # first degradation
        lq = self.compressor._apply_random_compression(self.jpeg_1(self.noise_1(self.resize_2(self.blur_1(gt_usm_copy, kernel1)))))

        # second degradation
        # lq = self.compressor._apply_random_compression(self.jpeg_2(self.noise_2(self.resize_2(self.blur_2(lq, kernel2), ori_h,ori_w)), ori_h,ori_w, sinc_kernel))

        return lq, gt_usm, kernel1, kernel2, sinc_kernel

    @torch.no_grad()
    def forward(self, img_gt, uint8=False):
        # read hwc 0-1 numpy
        # img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # augment
        # img_gt = self.augment(img_gt, True, True)
        # numpy 0-1 hwc -> tensor 0-1 chw
        # img_gt = self.np2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # add batch
        img_gt = img_gt.unsqueeze(0)
        img_gt_copy = img_gt.clone()

        # degradation_piepline
        lq, gt_usm, kernel1, kernel2, sinc_kernel = self.forward_deg(img_gt_copy)
        # clamp and round
        lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.
        # print(f'before crop: gt:{img_gt_copy.shape}, lq:{lq.shape}')
        # random crop
        # (gt, gt_usm), lq = self.paired_random_crop([img_gt_copy, gt_usm], lq, self.gt_size, self.scale)
        # print(f'after crop: gt:{gt.shape}, lq:{lq.shape}')

        # if uint8:
        #     gt, gt_usm, lq = self.tensor2np([gt, gt_usm, lq])
        #     return gt, gt_usm, lq, kernel1, kernel2, sinc_kernel

        return lq, gt_usm  # gt, kernel1, kernel2, sinc_kernel


    def blur_1(self, img, kernel1):
        img = filter2D(img, kernel1)
        return img

    def blur_2(self, img, kernel2):
        if np.random.uniform() < self.second_blur_prob:
            img = filter2D(img, kernel2)
        return img

    def resize_1(self, img):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(img, scale_factor=scale, mode=mode)

        return img

    def resize_2(self, img, ori_h, ori_w):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(
            img, size=(int(ori_h / self.scale * scale), int(ori_w / scale * scale)), mode=mode)

        return img

    def noise_1(self, img):
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob:
            img = random_add_gaussian_noise_pt(img, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def noise_2(self, img):
        gray_noise_prob = self.gray_noise_prob2
        if np.random.uniform() < self.gaussian_noise_prob2:
            img = random_add_gaussian_noise_pt(
                img, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def jpeg_1(self, img):
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*self.jpeg_range)
        img = torch.clamp(img, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        img = self.jpeger(img, quality=jpeg_p)
        return img

    def jpeg_2(self, out, ori_h, ori_w, sinc_kernel):
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)

        return out

    def generate_first_kernel(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel)

    def generate_second_kernel(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel2)

    def generate_sinc_kernel(self):
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        return sinc_kernel

    def np2tensor(self, imgs, bgr2rgb=False, float32=True):
        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            return img

        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        else:
            return _totensor(imgs, bgr2rgb, float32)

    def tensor2np(self, imgs):
        def _tonumpy(img):
            img = img.data.cpu().numpy().squeeze(0).transpose(1,2,0) #.astype(np.float32)
            img = np.uint8((img.clip(0,1) * 255.).round())
            return img

        if isinstance(imgs, list):
            return [_tonumpy(img) for img in imgs]
        else:
            return _tonumpy(imgs)


    def augment(self, imgs, hflip=True, rotation=True, flows=None, return_status=False):
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        return imgs

    def paired_random_crop(self, img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
        """Paired random crop. Support Numpy array and Tensor inputs.

        It crops lists of lq and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            gt_patch_size (int): GT patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth. Default: None.

        Returns:
            list[ndarray] | ndarray: GT images and LQ images. If returned results
                only have one element, just return ndarray.
        """

        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]

        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

        if input_type == 'Tensor':
            h_lq, w_lq = img_lqs[0].size()[-2:]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            h_lq, w_lq = img_lqs[0].shape[0:2]
            h_gt, w_gt = img_gts[0].shape[0:2]
        lq_patch_size = gt_patch_size // scale

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                             f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                             f'({lq_patch_size}, {lq_patch_size}). '
                             f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch
        if input_type == 'Tensor':
            img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
        else:
            img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        if input_type == 'Tensor':
            img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        else:
            img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        return img_gts, img_lqs


# init degradation process
degradation = Degradation(scale=4, gt_size=(480, 720))

def degradation_process(video_array):
    _, _, t, _, _ = video_array.shape
    # preprocess video
    video_array = video_array.to(torch.float32).cpu()
    video_array = (video_array + 1) * 0.5 # [-1, 1] -> [0, 1]
    video_array = rearrange(video_array, "B C T H W -> (B T) C H W")
    assert torch.max(video_array) <= 1 and torch.min(video_array) >= 0, "Values are NOT within [0, 1]."

    # degrade
    lq_list = []
    gt_list = []
    for video in video_array:
        lq, gt = degradation(video)
        lq_list.append(lq)
        gt_list.append(gt)

    lq = torch.cat(lq_list, dim=0)
    gt = torch.cat(gt_list, dim=0)
    lq = lq.clip(0, 1) * 2 - 1
    gt = gt.clip(0, 1) * 2 - 1
    lq = rearrange(lq, "(B T) C H W -> B C T H W", T=t).to(torch.float32)
    gt = rearrange(gt, "(B T) C H W -> B C T H W", T=t).to(torch.float32)
    return lq, gt