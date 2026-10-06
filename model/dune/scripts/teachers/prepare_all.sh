if [ "$#" -ne 1 ]; then
    echo "Usage: bash _prepare_all.sh MODELS_ROOT_DIR"
    exit 1
fi

MODELS_ROOT_DIR=${1}

umask 000

bash dinov2.sh ${MODELS_ROOT_DIR}
bash mast3r.sh ${MODELS_ROOT_DIR} # gcc-toolset-9 needed
bash multihmr.sh ${MODELS_ROOT_DIR}