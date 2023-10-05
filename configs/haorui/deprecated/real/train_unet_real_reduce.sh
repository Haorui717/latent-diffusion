nohup python main.py --base configs/haorui/unet_seg/real/train_unet_10.yaml --logdir logs/segmentation/real_reduce -t --gpus 4,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_20.yaml --logdir logs/segmentation/real_reduce -t --gpus 4,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_30.yaml --logdir logs/segmentation/real_reduce -t --gpus 4,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_40.yaml --logdir logs/segmentation/real_reduce -t --gpus 4,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_50.yaml --logdir logs/segmentation/real_reduce -t --gpus 5,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_60.yaml --logdir logs/segmentation/real_reduce -t --gpus 5,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_70.yaml --logdir logs/segmentation/real_reduce -t --gpus 5,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_80.yaml --logdir logs/segmentation/real_reduce -t --gpus 5,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/real/train_unet_90.yaml --logdir logs/segmentation/real_reduce -t --gpus 5,\
      > nohup_log/segmentation/real_reduce/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
