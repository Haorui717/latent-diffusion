#%%
# import matplotlib.pyplot as plt

# x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# y = [0.8354, 0.8516, 0.8655, 0.8565, 0.8543, 0.8646, 0.8611, 0.8704, 0.8770]

# # Create the plot
# plt.plot(x, y, marker='o')  # 'o' means it will also plot points at each (x,y) position

# # Add title and labels
# plt.title("Mean distance variation with training images")
# plt.xlabel("# training images")
# plt.ylabel("mean distance")

# # Display the plot
# plt.grid(True)
# plt.show()
#%%
# import os

# image_files = [_ for _ in os.listdir('/home/zongwei/haorui/ccvl15/haorui/datasets/__2023-08-11synthesized_polyps_merged/images')]
# mask_files = [_ for _ in os.listdir('/home/zongwei/haorui/ccvl15/haorui/datasets/__2023-08-11synthesized_polyps_merged/masks')]

# # the images and masks are not corresponding to each other
# # find the exception in the mask files that are not in the image files and vice versa

# exception_image_files = []
# for image_file in image_files:
#     if image_file not in mask_files:
#         exception_image_files.append(image_file)
# exception_mask_files = []
# for mask_file in mask_files:
#     if mask_file not in image_files:
#         exception_mask_files.append(mask_file)


#%% 

import sys
import io
from contextlib import contextmanager
from tqdm import tqdm

@contextmanager
def NO_OUTPUT():
    # Store the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout and stderr to a null device
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# Usage
with NO_OUTPUT():
    for i in tqdm(range(10000)):
        print("This will not be printed to the console.")
        a = 1 / 0

print("This will be printed to the console.")