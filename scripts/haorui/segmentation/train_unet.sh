nohup python main.py --base configs/haorui/unet_seg/train_unet.yaml --logdir logs/segmentation -t --gpus 0, > nohup_log/segmentation/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &