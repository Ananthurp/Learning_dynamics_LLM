#!/bin/bash
# ============================================================================
# Environment Setup Script for Learning Dynamics of LLM Finetuning
# ============================================================================
# Target System: NVIDIA RTX PRO 6000 (97GB VRAM) with CUDA 13.0
# This script creates a conda environment optimized for your hardware.
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Learning Dynamics LLM - Environment Setup"
echo "=============================================="

# Configuration
ENV_NAME="learntune"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. GPU support may not work."
fi

# Create conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
# For CUDA 12.x (compatible with CUDA 13.0 driver)
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install transformers==4.36.2
pip install datasets==2.16.1
pip install tokenizers==0.15.0
pip install accelerate==0.25.0

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install numpy
pip install tqdm
pip install wandb
pip install hydra-core==1.3.2
pip install omegaconf
pip install beautifulsoup4
pip install fsspec
pip install ipykernel
pip install jupyter
pip install matplotlib
pip install seaborn
pip install pandas

# Install tensor-parallel (optional, for multi-GPU tensor parallelism)
echo ""
echo "Installing tensor-parallel..."
pip install tensor-parallel

# Install flash-attention for faster training (optional but recommended)
echo ""
echo "Installing flash-attention (this may take a while)..."
pip install ninja packaging
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn installation failed. Continuing without it."

# Verify installation
echo ""
echo "=============================================="
echo "Verifying Installation..."
echo "=============================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')"
python -c "import wandb; print(f'Wandb version: {wandb.__version__}')"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test GPU memory allocation, run:"
echo "  python -c \"import torch; x = torch.zeros(1000, 1000, 1000).cuda(); print('GPU memory test passed!')\""
echo ""
