import cv2
import numpy as np
import torch

def rescale_tensor(tensor):
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    tensor = (tensor - min_val) / (max_val - min_val) * 255.0
    tensor = tensor.clamp(0, 255)
    return tensor

def compute_optical_flow(video_tensor):
    B, C, T, _, _ = video_tensor.shape
    assert C == 3, "Input video tensor must have 3 channels (RGB)."

    video_tensor = rescale_tensor(video_tensor).float()

    forward_flow = []
    backward_flow = []

    for b in range(B):
        forward_flow_batch = []
        backward_flow_batch = []
        for t in range(T - 1):
            frame1 = video_tensor[b, :, t, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            frame2 = video_tensor[b, :, t + 1, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            flow_forward = cv2.calcOpticalFlowFarneback(
                frame1_gray, frame2_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            forward_flow_batch.append(flow_forward)

            flow_backward = cv2.calcOpticalFlowFarneback(
                frame2_gray, frame1_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            backward_flow_batch.append(flow_backward)

        forward_flow_batch = np.stack(forward_flow_batch, axis=0)  # [T-1, H, W, 2]
        backward_flow_batch = np.stack(backward_flow_batch, axis=0)  # [T-1, H, W, 2]

        forward_flow.append(forward_flow_batch)
        backward_flow.append(backward_flow_batch)

    forward_flow = np.stack(forward_flow, axis=0)  # [B, T-1, H, W, 2]
    backward_flow = np.stack(backward_flow, axis=0)  # [B, T-1, H, W, 2]

    return torch.tensor(forward_flow).permute(0, 4, 1, 2, 3), torch.tensor(backward_flow).permute(0, 4, 1, 2, 3)