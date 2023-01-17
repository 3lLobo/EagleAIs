from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


HUB_NAME = "Intel/dpt-large"

feature_extractor = DPTFeatureExtractor.from_pretrained(HUB_NAME)
model = DPTForDepthEstimation.from_pretrained(HUB_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dpt_depth(image: np.ndarray) -> np.ndarray:
    """Depth estimation with DPT.

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Depth map
    """

    # prepare image for the model
    with torch.cuda.device(0):
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        gpumodel = model.to(device)
        with torch.no_grad():
            outputs = gpumodel(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
    output = prediction.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")

    return output


if __name__ == "__main__":
    from utils.image_utils import load_image, plot_comparison
    import os

    folder_path = "data/0024mod/"
    depth_folder = "res/DPT/"
    if not os.path.exists(depth_folder):
        os.mkdir(depth_folder)

    for image_path in os.listdir(folder_path):
        full_path = folder_path + image_path
        image = load_image(full_path)
        depthmap = dpt_depth(image)
        save_path = image_path.replace(".png", "_depth.png")
        plt.imsave(depth_folder + save_path, depthmap, cmap="gray")
        plot_comparison(image, depthmap, depth_folder + image_path)
