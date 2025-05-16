TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root data/ \
    --val_freq 1 --batch_size 12 --save_freq 5 --print_freq 1000 \
    --max_epoch 200 \
    --lr_backbone=1e-4 --lr=1e-4 \
    --dataset waymo-multi --test_dataset waymo-multi \
    --detect_intermediate --joint_det \
    --lr_decay_epochs 75 80 \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/ours \
    --self_attend --augment_det