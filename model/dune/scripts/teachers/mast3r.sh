# Publication title          : Grounding Image Matching in 3D with MASt3R
# Publication URL            : https://arxiv.org/abs/2406.09756
# Official Github repo       : https://github.com/naver/mast3r

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

pwd=${PWD}
source ./utils.sh

for arch in "vitlarge16";  do

    if [[ ${arch} == "vitlarge16" ]]; then
        # link for the pretrained model
        model_url="https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing Mast3r - ${arch}"
    model_dir=${root_dir}/mast3r/${arch}
    download_model ${model_url} ${model_dir}
done

# download the code, will be used in teachers/vit_master.py
cd ${root_dir}/mast3r
git clone --recursive https://github.com/naver/mast3r code

# compile the cuda kernels for RoPE
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd code/dust3r/croco/models/curope/
# do not forget to call before:
# - scl enable gcc-toolset-9 bash
echo "==> Compiling cuda kernels for RoPE"
python setup.py build_ext --inplace

cd ${pwd}