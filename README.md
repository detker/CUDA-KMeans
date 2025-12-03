# CUDA K-Means Clustering

./KMeans bin gpu1 ../../CUDA-KMeans/data/points_5mln_4d_5c.dat ../out.txt



A GPU-accelerated implementation of the K-Means clustering algorithm using CUDA.

## Prerequisites

### CUDA Toolkit
- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with compute capability 3.0 or higher
- NVIDIA driver installed

NOTE: System dependencies installation was tested only on Ubuntu distribution!

## Building the Project

### 1. Clone project repository
```bash
git clone https://github.com/detker/CUDA-KMeans
cd CUDA-KMeans
```

### 2. Getting the environment ready

#### Ubuntu/Debian

##### [Optional] Run project in Docker
```bash
# Add NVIDIA’s repository
curl -s -L https://nvidia.github.io/nvidia-container-toolkit/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

```bash
docker pull nvidia/cuda:12.3.1-devel-ubuntu22.04
```

```bash
sudo docker run -it --gpus all \
    --device=/dev/dri \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v ./CUDA-KMeans:/app/kmeans \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --network=host \
    nvidia/cuda:12.3.1-devel-ubuntu22.04 bash
```

and navigate to ```app/kmeans``` directory inside the container


##### System Dependencies

```bash
# Update package list
apt update

# Install all required dependencies
apt install -y \
    g++-9 \
    cmake \
    libglfw3-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libglx-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxxf86vm-dev
```


##### System Dependencies for Arch Linux

```bash
# Install required dependencies
sudo pacman -S \
    gcc \
    cmake \
    glfw-x11 \
    glew \
    mesa \
    libgl \
    glu \
    libx11 \
    libxrandr \
    libxinerama \
    libxcursor \
    libxi \
    libxxf86vm
```



### 3. Create build directory

```bash
mkdir build
cd build
```

### 4. Configure with CMake

```bash
cmake ..
```

If CMake cannot find CUDA automatically, specify the CUDA path:

```bash
cmake -DCUDAToolkit_ROOT=/usr/local/cuda ..
```

### 5. Build

```bash
make
```
