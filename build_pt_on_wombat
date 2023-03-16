# Instalation script for wombat
# I used it to build PT on wombat in early 2022
# PyTorch, Horovod is installed from sources


# set up modules  on wombat
    module purge
    module load cuda
    export CUDA_HOME=/opt/software/cuda-11.2.0


    conda create --prefix /data/d10a/users/boyda/.conda/envs/torch-02-22 python=3.9
    conda activate torch-02-22


# additional packages for flow-based
    conda install pymongo tqdm numpy h5py -y
    pip install sacred scipy


# cuda packages (and cmake) needed for pytorch, horovod, cupy,...
# we will use local cudatoolkit but we install it here to allow conda deduction of version of cudnn nccl etc.
    conda install -c conda-forge -c pytorch cudatoolkit=11.2 cudnn nccl magma-cuda112 -y
    pip install cupy-cuda112


# MPI4PY

# CUDA-aware Open MPI is needed for mpi4py for support operations on GPUs
# and (possibly) for DDP, DeepZero since they use it
# if using local build it must be compiles cuda-aware, e.g. with ucx
# # example of instalation in this case
# #MPI_DIR=/opt/software/openmpi-...
# #export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
# #CC=$(which mpicc) CXX=$(which mpicxx)  pip install --no-cache-dir mpi4py

# package openmpi from conda-forge is compiled with cuda-aware but it is disable by defaule
# such that OMPI_MCA_opal_cuda_support=true must be used for mpirun
    conda install -c conda-forge mpi4py openmpi
# set cuda-aware option to true by defaule - change value opal_cuda_support = 1
# at vim d10/.conda/envs/torch-02-22/etc/openmpi-mca-params.conf

# test mpi4py with cupa support
    mpirun -n 2 --mca btl_base_verbose 1 python test_mpi4py.py


# Building pytorch

# packages for build
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
# if cmake is old
# conda install -c conda-forge cmake">=3.13" -y

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# sms for rtx2080ti, A100 and rtx1080ti
TORCH_CUDA_ARCH_LIST="7.5;8.0;6.1" python setup.py install

# check build
cd ..
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.cuda.is_available()); print(torch.distributed.is_available())"


# Building horovod

cd ..
git clone --recursive https://github.com/horovod/horovod
cd horovod
HOROVOD_NCCL_HOME=$CONDA_PREFIX  HOROVOD_CUDA_HOME=$CUDA_HOME HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_NCCL_LINK=SHARED  python setup.py bdist_wheel
cd ..
HVD_WHL=$(find horovod/dist/ -name "horovod*.whl" -type f)
pip install software/$HVD_WHL

# check build
horovodrun -cb

# Problems
# if scipy errors on libc lib -> install new gcc
