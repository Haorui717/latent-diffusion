nohup python main.py --base configs/haorui/unet_seg/train_resunetpp_onthefly/train_resunetpp_onthefly_10.yaml --resume /home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/logs/segmentation/train_resunetpp_onthefly/2023-09-14T16-24-42_train_resunetpp_onthefly_10 \
      --logdir logs/segmentation/train_resunetpp_onthefly -t --gpu 0,1 > nohup_log/segmentation/train_resunetpp_onthefly/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/train_resunetpp_onthefly/train_resunetpp_onthefly_20.yaml --resume /home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/logs/segmentation/train_resunetpp_onthefly/2023-09-14T16-24-47_train_resunetpp_onthefly_20 \
      --logdir logs/segmentation/train_resunetpp_onthefly -t --gpu 2,3 > nohup_log/segmentation/train_resunetpp_onthefly/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/train_resunetpp_onthefly/train_resunetpp_onthefly_900.yaml --resume /home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/logs/segmentation/train_resunetpp_onthefly/2023-09-14T16-24-52_train_resunetpp_onthefly_900 \
      --logdir logs/segmentation/train_resunetpp_onthefly -t --gpu 4,5 > nohup_log/segmentation/train_resunetpp_onthefly/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
