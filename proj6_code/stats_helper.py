import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Tip: You can use any function you want to find mean and standard deviation

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  pixel_sum = 0
  pixel_sum_squared = 0
  pixel_count = 0

  # Traverse through all images in the directory
  for root, _, files in os.walk(dir_name):
    for file in files:
      if not file.endswith('.jpg'):
        continue
      # Full path of the image
      img_path = os.path.join(root, file)

      try:
        # Open image and convert to grayscale
        img = Image.open(img_path).convert('L')

        # Normalize pixel values to [0, 1]
        img = np.array(img) / 255.0

        # Update pixel statistics
        pixel_sum += np.sum(img)
        pixel_sum_squared += np.sum(img ** 2)
        pixel_count += img.size  # Total number of pixels
      except Exception as e:
        print(f"Error processing image {img_path}: {e}")

  # Compute mean and standard deviation
  mean = pixel_sum / pixel_count
  std = np.sqrt((pixel_sum_squared / pixel_count) - (mean ** 2))
  mean = np.array([mean])
  std = np.array([std])
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
