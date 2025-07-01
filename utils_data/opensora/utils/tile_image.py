import cv2
import torch
import numpy as np
from einops import rearrange

def block_image(image, block_size, overlap):
    image = rearrange(image, "C H W -> H W C")
    height, width, _ = image.shape
    block_images = []

    # 计算重叠的像素数
    overlap_pixels = int(block_size * overlap)

    # 逐行遍历图像
    for y in range(0, height, block_size - overlap_pixels):
        for x in range(0, width, block_size - overlap_pixels):
            # 确保块的尺寸一致，填充超出边界的部分
            block = np.zeros((block_size, block_size, 3), dtype=image.dtype)
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block[:y_end - y, :x_end - x] = image[y:y_end, x:x_end]
            block = rearrange(block, "H W C -> C H W")
            block_images.append(block)

    return block_images

def combine_blocks(blocks, image_shape, block_size, overlap):
    height, width, _ = image_shape
    overlap_pixels = int(block_size * overlap)
    reconstructed_image = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
    weight_sum = torch.zeros((height, width, 3), dtype=torch.float32).cuda()

    # 生成高斯权重矩阵
    weights = _gaussian_weights(block_size, block_size, 1).squeeze().cpu().numpy()

    idx = 0
    for y in range(0, height, block_size - overlap_pixels):
        for x in range(0, width, block_size - overlap_pixels):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = torch.tensor(blocks[idx], dtype=torch.float32).cuda()

            # 为块生成相应的权重矩阵
            block = rearrange(block, "C H W -> H W C")
            block_height, block_width = block.shape[:2]
            weight = torch.tensor(weights[:block_height, :block_width], dtype=torch.float32).unsqueeze(-1).cuda()
            weight = weight.expand(-1, -1, 3)  # Expand weight to match the number of channels
            
            # Adjust the dimensions of weight if necessary
            reconstructed_image[y:y_end, x:x_end, :] += block[:y_end - y, :x_end - x] * weight[:y_end - y, :x_end - x]
            weight_sum[y:y_end, x:x_end, :] += weight[:y_end - y, :x_end - x]
            idx += 1

    weight_sum[weight_sum == 0] = 1.0
    reconstructed_image /= weight_sum

    return reconstructed_image

def _gaussian_weights(tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    var = 0.01
    midpoint_w = (tile_width - 1) / 2
    x_probs = [np.exp(-(x - midpoint_w) * (x - midpoint_w) / (tile_width * tile_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
               for x in range(tile_width)]
    midpoint_h = (tile_height - 1) / 2
    y_probs = [np.exp(-(y - midpoint_h) * (y - midpoint_h) / (tile_height * tile_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
               for y in range(tile_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(nbatches, 1, 1, 1)