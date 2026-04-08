import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class EDABase:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        sns.set_theme(style="whitegrid")

    def calculate_brightness(self, image_path):
        """Calculates the average brightness of an image using the V channel of HSV."""
        img = cv2.imread(image_path)
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 2].mean()
        return None

    def save_plot(self, plt_obj, filename):
        """Saves the current plot to the reports directory."""
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, filename)
        plt_obj.savefig(path)
        print(f"Plot saved to {path}")
