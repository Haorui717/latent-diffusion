CUDA_VISIBLE_DEVICES=2, nohup python scripts/haorui/reduce_data_scripts/models_comparison.py \
--base /home/zongwei/haorui/ccvl15/haorui/latent-diffusion-Laptop/latent-diffusion/scripts/haorui/reduce_data_scripts/models_comparison.yaml \
--image_path /home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/PolyGen/PolyGEN_n_all_images.txt \
--mask_path /home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/PolyGen/PolyGen_CVC_masks.txt \
--batch_size 48 \
--outdir /home/zongwei/haorui/ccvl15/haorui/latent-diffusion-Laptop/latent-diffusion/outputs/gen_samples \
--mask_shuffle \
--nums 10000 \
--random_image \
> nohup_log/reduce_data_logs/gen_samples/log_$(date +"%Y:%m:%d:%H:%M:%S").txt 2>&1 &