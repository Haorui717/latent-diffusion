nohup python main.py --base configs/haorui/unet_seg/resunetpp/train_resunetpp_900.yaml --logdir logs/segmentation/resunetpp -t --gpu 3,4\
      --resume /home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/logs/segmentation/resunetpp/2023-09-13T17-43-51_train_resunetpp_900  > nohup_log/segmentation/resunetpp/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/resunetpp/train_resunetpp_10.yaml  --logdir logs/segmentation/resunetpp -t --gpu 5,6\
      --resume /home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/logs/segmentation/resunetpp/2023-09-13T17-43-56_train_resunetpp_10   > nohup_log/segmentation/resunetpp/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &