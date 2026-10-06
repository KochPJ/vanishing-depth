# Publication title          : Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot
# Publication URL            : https://arxiv.org/abs/2402.14654
# Official Github repo       : https://github.com/naver/multi-hmr

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

source ./utils.sh

for arch in "vitlarge14_672";  do

    if [[ ${arch} == "vitlarge14_672" ]]; then
        model_url="https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L.pt"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing Multi-HMR - ${arch}"
    model_dir=${root_dir}/multihmr/${arch}
    download_model ${model_url} ${model_dir}

done