# model configs
load_dav2=1
resolution=3
pred_confidence_input=1
multi_resolution_learnable_gradients_weights="uniform"
optim_layer_input_clamp=1.0
depth_activation_format='exp'
max_depth=300.0
whiten_sparse_depths=1
backbone='rgbd'

# checkpoints
ckpt=../checkpoints/modelv1.1_best_72epochs.pt

dir=$(dirname $ckpt)
base=$(basename $dir)
log_dir="../experiments/test_${base}"

echo $log_dir
mkdir $log_dir


# void
for data_name in VOID_sample1500 VOID_sample500 VOID_sample150
do
  python main.py \
    --dir_data ../datasets/uniformat_release/"${data_name}" --val_data_name Uniformat \
    --max_depth $max_depth --data_normalize_median 1 \
    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
    --load_dav2 $load_dav2 \
    --gpus 0 \
    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
    --log_dir "$log_dir/" \
    --save "val_${data_name}" \
    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
    --pretrain $ckpt --test_only
done
