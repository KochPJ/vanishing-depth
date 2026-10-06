# Publication title          : DINOv2: Learning Robust Visual Features without Supervision
# Publication URL            : https://arxiv.org/abs/2304.07193
# Official Github repo       : https://github.com/facebookresearch/dinov2

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

source ./utils.sh

for arch in "vitlarge14_reg"; do

    if [[ ${arch} == "vitsmall14" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    elif [[ ${arch} == "vitbase14" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    elif [[ ${arch} == "vitgiant14" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
    elif [[ ${arch} == "vitlarge14_reg" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
    elif [[ ${arch} == "vitgiant14_reg" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing DINO-v2 - ${arch}"
    model_dir=${root_dir}/dinov2/${arch}
    download_model ${model_url} ${model_dir}

done