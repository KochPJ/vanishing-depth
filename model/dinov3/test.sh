PYTHONPATH=${PWD} python -m dinov3/run.submit dinov3/eval/log_regression.py \
  model.config_file=/home/kochpaul/git/vanishing-depth-self-supervised/data/dinov3res/config.yaml \
  model.pretrained_weights=/home/kochpaul/git/vanishing-depth-self-supervised/data/dinov3res/teacher_checkpoint.pth \
  output_dir=/home/kochpaul/git/vanishing-depth-self-supervised/data/dinov3res \
  train.dataset=ImageNet:split=TRAIN:root=/home/kochpaul/git/vanishing-depth-self-supervised/data/imagenet:extra=/home/kochpaul/git/vanishing-depth-self-supervised/data/imagenet \
  eval.test_dataset=ImageNet:split=VAL:root=/home/kochpaul/git/vanishing-depth-self-supervised/data/imagenet:extra=/home/kochpaul/git/vanishing-depth-self-supervised/data/imagenet