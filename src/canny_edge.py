import cv2
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import os
import sys
from PIL import Image


def canny_edge(img: np.ndarray, low_threshold: int, high_threshold: int) -> Image:
    """Detect edges in an image using the canny edge detection algorithm.

    Args:
        img (np.ndarray): Image as array to detect edges on.
        low_threshold (int): Low threshold for hysteresis.
        high_threshold (int): High threshold for hysteresis.

    Returns:
        Image: Image with edges detected.
    """
    # Gaussian blur
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Erode
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.Canny(edges, low_threshold, high_threshold)

    return edges


if __name__ == "__main__":
    img_path = "/home/wolf/worqspace/EagleEyez/data/png/s1/0016.png"
    img = Image.open(img_path)
    img = np.array(img)
    edges = canny_edge(img, 100, 330)
    Image.fromarray(edges).show()
