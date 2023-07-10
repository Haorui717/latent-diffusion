import os
import shutil


mask_paths = [
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C1/masks_C1',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C2/masks_C2',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/masks_C3',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C4/masks_C4',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C5/masks_C5',
]

# images name ends with <imagename>.jpg/.png but masks name ends with <imagename>_mask.jpg/.png
# so we need to rename the masks by removing the '_mask' part

for mask_path in mask_paths:
    for filename in os.listdir(mask_path):
        if '_mask' in filename:
            os.rename(os.path.join(mask_path, filename), os.path.join(mask_path, filename.replace('_mask', '')))

