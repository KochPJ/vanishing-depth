download_model () {
    url=${1}
    dir=${2}

    mkdir -p ${dir}
    echo "==> Model directory: ${dir}"

    cd ${dir}
    wget -q -O model.pth ${url}
    model_ckpt="${dir}/model.pth"
    if [[ ! -f "${model_ckpt}" ]]; then
        echo "==> Couldn't download model checkpoint, please see the bash script for possible further instructions."
    else
        echo "==> Model checkpoint: ${model_ckpt}"
    fi
}