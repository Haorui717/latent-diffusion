nohup python main.py --base configs/haorui/unet_seg/resunetpp/train_resunetpp_1800.yaml  --logdir logs/segmentation/resunetpp -t --gpu 1,2\
        > nohup_log/segmentation/resunetpp/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/resunetpp/train_resunetpp_1000.yaml  --logdir logs/segmentation/resunetpp -t --gpu 3,5\
        > nohup_log/segmentation/resunetpp/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &
sleep 5
nohup python main.py --base configs/haorui/unet_seg/resunetpp/train_resunetpp_500.yaml   --logdir logs/segmentation/resunetpp -t --gpu 6,7\
        > nohup_log/segmentation/resunetpp/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &