import os
import shutil

# image_paths = [
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C1/images_C1',
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C2/images_C2',
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/images_C3',
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C4/images_C4',
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C5/images_C5',
#     '/home/zongwei/haorui/ccvl15/haorui/datasets/CVC-ClinicDB/PNG/Original',
# ]

mask_paths = [
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C1/masks_C1',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C2/masks_C2',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/masks_C3',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C4/masks_C4',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C5/masks_C5',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/CVC-ClinicDB/PNG/Ground Truth',
]

os.makedirs('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset', exist_ok=True)

# image_list = []
mask_list = []

# for image_path in image_paths:
#     for filename in os.listdir(image_path):
#         image_list.append(os.path.join(image_path, filename))

for mask_path in mask_paths:
    for filename in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, filename))

# image_list.sort()
mask_list.sort()

# write image list to txt file
# with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/PolyGen_images.txt', 'w') as f:
#     for item in image_list:
#         f.write("%s\n" % item)

# write mask list to txt file
with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/PolyGen_masks.txt', 'w') as f:
    for item in mask_list:
        f.write("%s\n" % item)


