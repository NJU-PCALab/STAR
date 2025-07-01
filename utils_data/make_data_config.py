# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=32,
    frame_interval=2,
    image_size=(720, 1280),
)

data_path = ''
save_path = ''
dtype = "bf16"
num_workers = 2
batch_size = 1  # now only support batch_size=1
seed = 42