conda_dir="/path/to/conda" # Change this to your conda installation path
conda_env_name="dune" # Change this to your conda environment name

echo "----------------------------------------------------------------------------------------------------"
hostnamectl
echo "----------------------------------------------------------------------------------------------------"
nvidia-smi
echo "----------------------------------------------------------------------------------------------------"
free -g
echo "----------------------------------------------------------------------------------------------------"
df -h /dev/shm
echo "----------------------------------------------------------------------------------------------------"
df -h /local
echo "----------------------------------------------------------------------------------------------------"

check_python_pkg () {
    pkg=${1}
    if python -c "import ${pkg}" &> /dev/null; then
        ver=$(python -c "import ${1}; print(${1}.__version__)")
        printf "%-30s : %s\n" "${1} version" "${ver}"
    fi
}

check_git () {
    if [ -d .git ]; then
        printf "%-30s : %s\n" "git commit SHA-1" "$(git rev-parse HEAD)"
    else
        printf "This code is not in a git repo\n"
    fi;
}

echo "--------------------------------------------------"
source /etc/proxyrc
source ${conda_dir}/bin/activate base
conda activate ${conda_env_name}
export LD_LIBRARY_PATH=${conda_dir}/envs/${conda_env_name}/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH="${PWD}:${PYTHONPATH}"

printf "%-30s : %s\n" "conda environment name" "${CONDA_DEFAULT_ENV}"
printf "%-30s : %s\n" "conda environment path" "${CONDA_PREFIX}"
check_python_pkg "torch"
check_python_pkg "torchvision"
check_python_pkg "timm"
check_python_pkg "PIL"
check_python_pkg "mmcv"
printf "%-30s : %s\n" "libjpeg-turbo support" "$(python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))")"
num_cores="$(python -c "import os; print(len(os.sched_getaffinity(0)))")"
printf "%-30s : %s\n" "Number of processors:" "${num_cores}"
check_git

export ONEDAL_NUM_THREADS=${num_cores}
export OMP_NUM_THREADS=${num_cores}
export MKL_NUM_THREADS=${num_cores}

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

echo "--------------------------------------------------"
echo "${SLURM_JOB_NUM_NODES} nodes available"
IFS=','; gpus=($CUDA_VISIBLE_DEVICES); unset IFS;
export N_GPUS=${#gpus[@]}
echo "${N_GPUS} GPU(s) available on this node"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"