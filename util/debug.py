from PIL import Image
import numpy as np
import cv2

# load image using cv2
cv2_img = cv2.imread('/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/masks_C3/C3_EndoCV2021_0048_mask.jpg')

# load image using PIL
pil_img = Image.open('/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/masks_C3/C3_EndoCV2021_0048_mask.jpg').convert('L')

# convert cv2 image to numpy array
cv2_img = np.array(cv2_img).astype(np.float32) / 255.0

# convert PIL image to numpy array
pil_img = np.array(pil_img).astype(np.float32) / 255.0

# print the shape of cv2 image and PIL image
print(cv2_img.shape)
print(pil_img.shape)
