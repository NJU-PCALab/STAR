pip3 install --upgrade pip

pip3 install greenlet==1.1.3
pip3 install gevent==22.8.0
pip3 install torch 
pip3 install torchvision

pip3 install ftfy
pip3 install numpy
pip3 install tqdm
pip3 install psutil
pip3 install pre-commit
pip3 install rich
pip3 install click
pip3 install fabric
pip3 install contexttimer
pip3 install safetensors
pip3 install einops
pip3 install pydantic
pip3 install ray
pip3 install protobuf
pip3 install gdown
pip3 install pyav
pip3 install tensorboard
pip3 install timm
pip3 install matplotlib
pip3 install accelerate
pip3 install diffusers
pip3 install transformers
pip3 install ipdb
pip3 install opencv-python
pip3 install webdataset
pip3 install gateloop_transformer
pip3 install kornia
pip3 install scipy
sudo apt-get install -y libgl1-mesa-dev

# install flash attention (optional)
# set enable_flashattn=False in config to avoid using flash attention
pip3 install packaging
pip3 install ninja
pip3 install flash-attn --no-build-isolation

# install apex (optional)
# set enable_layernorm_kernel=False in config to avoid using apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
#pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
# cp -r /mnt/bn/videodataset/VSR/data/compile/xformers-0.0.25.post1-cp39-cp39-manylinux2014_x86_64.whl .
# pip install xformers-0.0.25.post1-cp39-cp39-manylinux2014_x86_64.whl

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
pip uninstall colossalai -y
pip install colossalai==0.3.7
cd ..