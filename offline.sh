
data_dir='/home/atkeonlab-3/Desktop/YiqiProject/dataset/PointMaze_UMaze-v3_debug.npz'

# remember to wandb login

python ./tdmpc2/train.py \
    task=pointmaze \
    model_size=48 \
    action_dim=2 \
    batch_size=1024 \
    data_dir=${data_dir} \
    enable_wandb=true \
    eval_freq=100 \
    wandb_project='debug_tdmpc2' \
    wandb_entity='wiki' 
