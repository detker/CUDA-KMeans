# CUDA K-Means Clustering

A GPU-accelerated implementation of the K-Means clustering algorithm using CUDA.

## Prerequisites

### CUDA Toolkit
- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with compute capability 3.0 or higher
- NVIDIA driver installed

### System Dependencies

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install all required dependencies
sudo apt install -y \
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

#### Arch Linux

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

## Building the Project

### 1. Unzip and open project directory

### 2. Create build directory

```bash
mkdir build
cd build
```

### 3. Configure with CMake

```bash
cmake ..
```

If CMake cannot find CUDA automatically, specify the CUDA path:

```bash
cmake -DCUDAToolkit_ROOT=/usr/local/cuda ..
```

### 4. Build

```bash
make
```