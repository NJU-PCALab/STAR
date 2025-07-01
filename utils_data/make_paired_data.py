import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module, DATASETS
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype

import torch.nn.functional as F
from einops import rearrange
from opensora.datasets.high_order.degrade_video import degradation_process
from tqdm import tqdm

def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    cfg.dataset['data_path'] = cfg.data_path
    dataset = build_module(cfg.dataset, DATASETS)
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    dataloader = prepare_dataloader(**dataloader_args)
    dataloader_iter = iter(dataloader)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir_gt = cfg.save_path + '/gt'
    save_dir_lq = cfg.save_path + '/lq'
    save_dir_txt = cfg.save_path + '/text'
    os.makedirs(save_dir_gt, exist_ok=True)
    os.makedirs(save_dir_lq, exist_ok=True)
    os.makedirs(save_dir_txt, exist_ok=True)

    # 4.1. batch generation with progress bar
    for _, batch in tqdm(enumerate(dataloader_iter), total=len(dataloader), desc="Processing 10K Batches"):
        x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W], HR-video
        fps = batch.pop('fps')

        # generate LR-video
        lr, x = degradation_process(x)
        _, _, t, _, _ = lr.shape
        lr = rearrange(F.interpolate(rearrange(lr, "B C T H W -> (B T) C H W"), scale_factor=4, mode='bicubic'), "(B T) C H W -> B C T H W", T=t)
        y = batch.pop("text")

        # 4.4. save samples
        if not use_dist or coordinator.is_master():
            for i in range(0, lr.shape[0]):
                save_dir_gt_ = os.path.join(save_dir_gt, f"{sample_idx}")
                save_dir_lq_ = os.path.join(save_dir_lq, f"{sample_idx}")
                save_dir_txt_ = os.path.join(save_dir_txt, f"{sample_idx}.txt")

                save_sample(x[i], fps=fps / cfg.dataset['frame_interval'], save_path=save_dir_gt_)
                save_sample(lr[i], fps=fps / cfg.dataset['frame_interval'], save_path=save_dir_lq_)
                with open(save_dir_txt_, 'w', encoding='utf-8') as file:
                    file.write(y[i])

                sample_idx += 1

if __name__ == "__main__":
    main()