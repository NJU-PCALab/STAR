#!/usr/bin/env python
# coding=utf-8

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import torch
import torch.nn.functional as F
import torch.fft
from typing import Tuple

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)

from video_to_video.modules import *
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger
from video_super_resolution.dataset import PairedCaptionVideoDataset

from diffusers import AutoencoderKLTemporalDecoder

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)

def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable

def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1 * 8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='NOTHING',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help=(
            "Length of each training video"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SeeSR",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    return args


args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)
logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be divided by the number of processes assuming batches are multiplied by the number of processes
)


###-------------------------------------
# Bulid Model
###------------------------------------

# text_encoder
text_encoder = FrozenOpenCLIPEmbedder(pretrained="laion2b_s32b_b79k")
text_encoder.requires_grad_(False)
logger.info(f'Build text encoder with CLIP')

# U-Net with ControlNet
model = ControlledV2VUNet()
load_dict = torch.load(args.pretrained_model_path, map_location='cpu')
if 'state_dict' in load_dict:
    load_dict = load_dict['state_dict']

incompatible_keys = model.load_state_dict(load_dict, strict=False)
logger.info('Load model path {}, with local status {}'.format(args.pretrained_model_path, incompatible_keys))
model_numel, model_numel_trainable = get_model_numel(model)
logger.info(
    f"Total model params: {format_numel_str(model_numel)}"
)

# Noise scheduler
sigmas = noise_schedule(
    schedule='logsnr_cosine_interp',
    n=1000,
    zero_terminal_snr=True,
    scale_min=2.0,
    scale_max=4.0)
noise_scheduler = GaussianDiffusion(sigmas=sigmas)
logger.info('Build noise_scheduler with GaussianDiffusion')

# Temporal VAE
vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
vae.eval()
vae.requires_grad_(False)
logger.info('Build Temporal VAE')

###-------------------------------------
# Bulid dataset & dataloader
###-------------------------------------
train_dataset = PairedCaptionVideoDataset(
    root_folders=[
        args.train_data_dir
    ], 
    num_frames=args.num_frames,
)

train_dataloader = torch.utils.data.DataLoader(
train_dataset,
num_workers=args.dataloader_num_workers,
batch_size=args.train_batch_size,
shuffle=False
)

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

###-------------------------------------
# Optimizer creation & lr_scheduler
###-------------------------------------
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

print(f'=================Optimize ControlNet ======================')

# For training VideoControlNet

params_to_optimize = set()

params_to_optimize.update(model.VideoControlNet.parameters())

for name, param in model.named_parameters():
    if 'local' in name:
        print(f'{name} will be optimized')
        params_to_optimize.add(param)

params_to_optimize = list(params_to_optimize)

# Calculate the total number of parameters
total_params = sum(param.numel() for param in params_to_optimize)
total_params_million = total_params / 1_000_000
print(f"Total number of trainable parameters to optimize: {total_params_million:.2f} million")


print(f'start to load optimizer...')

optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

# Prepare everything with `accelerator`.
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move vae and text_encoder to device and cast to weight_dtype
vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)

def tensor2latent(t, vae):
    video_length = t.shape[2]
    t = rearrange(t, "b c f h w -> (b f) c h w")
    chunk_size = 1
    latents_list = []
    for ind in range(0,t.shape[0],chunk_size):
        latents_list.append(vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
    latents = torch.cat(latents_list, dim=0)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents

def temporal_vae_decode(z, num_f):
    return vae.decode(z/vae.config.scaling_factor, num_frames=num_f).sample

def vae_decode_chunk(z, chunk_size=3):
    z = rearrange(z, "b c f h w -> (b f) c h w")
    video = []
    for ind in range(0, z.shape[0], chunk_size):
        num_f = z[ind:ind+chunk_size].shape[0]
        video.append(temporal_vae_decode(z[ind:ind+chunk_size],num_f))
    video = torch.cat(video)
    return video


def fourier_transform(x, balance=None):
    """
    Apply Fourier transform to the input tensor and separate it into low-frequency and high-frequency components.

    Args:
    x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
    balance (torch.Tensor or float, optional): Learnable balance parameter for adjusting the cutoff frequency.

    Returns:
    low_freq (torch.Tensor): Low-frequency components (with real and imaginary parts)
    high_freq (torch.Tensor): High-frequency components (with real and imaginary parts)
    """
    # Perform 2D Real Fourier transform (rfft2 only computes positive frequencies)
    x = x.to(torch.float32)
    fft_x = torch.fft.rfft2(x, dim=(-2, -1))
    
    # Calculate magnitude of frequency components
    magnitude = torch.abs(fft_x)

    # Set cutoff based on balance or default to the 80th percentile of the magnitude for low frequency
    if balance is None:
        # Downsample the magnitude to reduce computation for large tensors
        subsample_size = 10000  # Adjust based on available memory and tensor size
        if magnitude.numel() > subsample_size:
            # Randomly select a subset of values to approximate the quantile
            magnitude_sample = magnitude.flatten()[torch.randint(0, magnitude.numel(), (subsample_size,))]
            cutoff = torch.quantile(magnitude_sample, 0.8)  # 80th percentile for low frequency
        else:
            cutoff = torch.quantile(magnitude, 0.8)  # 80th percentile for low frequency
    else:
        # balance is clamped for safety and used to scale the mean-based cutoff
        cutoff = magnitude.mean() * (1 + 10 * balance)

    # Smooth mask using sigmoid to ensure gradients can pass through
    sharpness = 10  # A parameter to control the sharpness of the transition
    low_freq_mask = torch.sigmoid(sharpness * (cutoff - magnitude))
    
    # High-frequency mask can be derived from low-frequency mask (1 - low_freq_mask)
    high_freq_mask = 1 - low_freq_mask
    
    # Separate low and high frequencies using smooth masks
    low_freq = fft_x * low_freq_mask
    high_freq = fft_x * high_freq_mask

    # Return real and imaginary parts separately
    low_freq = torch.stack([low_freq.real, low_freq.imag], dim=-1)
    high_freq = torch.stack([high_freq.real, high_freq.imag], dim=-1)
    
    return low_freq, high_freq


def extract_frequencies(video: torch.Tensor, balance=None):
    """
    Extract high-frequency and low-frequency components of a video using Fourier transform.

    Args:
    video (torch.Tensor): Input video tensor of shape [batch_size, channels, frames, height, width]

    Returns:
    low_freq (torch.Tensor): Low-frequency components of the video
    high_freq (torch.Tensor): High-frequency components of the video
    """
    # batch_size, channels, frames, _, _ = video.shape
    video = rearrange(video, 'b c t h w -> (b t) c h w')  # Reshape for Fourier transform

    # Apply Fourier transform to each frame
    low_freq, high_freq = fourier_transform(video, balance=balance)

    return low_freq, high_freq

###-------------------------------------
# Train
###-------------------------------------
progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=0,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)
global_step = 0

for epoch in range(0, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        with accelerator.accumulate(model):
            video_data = batch.pop('gt').to(accelerator.device, dtype=weight_dtype) # [b, c, t, h, w]
            lq = batch.pop('lq').to(accelerator.device, dtype=weight_dtype)
            text = batch.pop('text')

            with torch.no_grad():
                # Process hq & lq video
                video_data_feature = tensor2latent(video_data, vae)
                lq_feature = tensor2latent(lq, vae)

                # Process text
                model_kwargs = {}
                model_kwargs['y'] = text_encoder(text).detach()

            # Diffusion process
            bsz = video_data_feature.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=video_data_feature.device)
            timesteps = timesteps.long()
            noise = torch.randn_like(video_data_feature)
            noised_video = noise_scheduler.diffuse(video_data_feature, timesteps, noise=noise)
            
            # == video meta info ==
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(accelerator.device, weight_dtype)
            
            model_kwargs['hint'] = lq_feature

            # Predict the velocity
            out = model(noised_video, timesteps, **model_kwargs)
            target = noise_scheduler.get_velocity(x0=video_data_feature, xt=noised_video, t=timesteps)
            
            # get the low-freq & high-freq from x0
            pred_x0 = noise_scheduler.get_x0(v=out, xt=noised_video, t=timesteps).to(accelerator.device, weight_dtype)
            # Learning the cutoff frequency
            with torch.no_grad():
                pred_x0 = vae_decode_chunk(pred_x0, chunk_size=3).permute(1, 0, 2, 3).unsqueeze(0)
            low_freq_pred_x0, high_freq_pred_x0 = extract_frequencies(pred_x0)
            low_freq_x0, high_freq_x0 = extract_frequencies(video_data)

            # v-prediction loss
            loss_v = F.mse_loss(out.float(), target.float(), reduction="mean")

            # timestep-aware loss
            alpha = 2
            ct = (timesteps/999) ** alpha
            loss_low = F.l1_loss(low_freq_pred_x0.float(), low_freq_x0.float(), reduction="mean")
            loss_high = F.l1_loss(high_freq_pred_x0.float(), high_freq_x0.float(), reduction="mean")
            loss_t = 0.01*(ct * loss_low + (1 - ct) * loss_high)

            # Calculate the loss & Backword
            beta = 1
            weight_t = 1 - timesteps/999
            loss = loss_v + beta * weight_t * loss_t
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = list(model.module.VideoControlNet.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

            # Save weights
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            

            logs = {"loss_high": loss_high.detach().item(), "loss_low": loss_low.detach().item(), "loss_v": loss_v.detach().item(), "total_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

accelerator.end_training()