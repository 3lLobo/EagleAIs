import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import io


HUB_NAME = "intel-isl/MiDaS"
midas_type = "DPT_Large"  # @param ["DPT_Large", "DPT_Hybrid"]
model = torch.hub.load(HUB_NAME, midas_type)

gpu_device = torch.device("cuda")
model.to(gpu_device)
model.eval()


transform = torch.hub.load(HUB_NAME, "transforms").dpt_transform


def midas_depth(image: np.ndarray) -> np.ndarray:
    """
    Depth estimation with MiDaS.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: depthmap
    """
    transformed_image = transform(image).to(gpu_device)

    with torch.no_grad():
        prediction = model(transformed_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.squeeze().cpu().numpy()

    return output


if __name__ == "__main__":
    import os
    from utils.image_utils import plot_comparison, load_image

    folder_path = "data/0024mod/"
    depth_folder = "res/MiDas/"
    if not os.path.exists(depth_folder):
        os.mkdir(depth_folder)

    for image_path in os.listdir(folder_path):
        full_path = folder_path + image_path
        image = load_image(full_path)
        depthmap = midas_depth(image)
        save_path = image_path.replace(".png", "_depth.png")
        plt.imsave(depth_folder + save_path, depthmap, cmap="gray")
        plot_comparison(image, depthmap, depth_folder + image_path)
