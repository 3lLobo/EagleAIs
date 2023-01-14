import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import io

# @title Load MiDaS
# @markdown Select the type of model you want.
# @markdown - DPT_Large: Probably the best choice. It's an extremely accurate model, but can be slow.
# @markdown - DPT_Hybrid: Faster than DPT_Large. Still quite an accurate model, but noticeably less so.
midas_type = "DPT_Large"  # @param ["DPT_Large", "DPT_Hybrid"]

model = torch.hub.load("intel-isl/MiDaS", midas_type)

# @markdown ### GPU Acceleration and Inference Mode
# @markdown We'll also put the model on the GPU to accelerate inference.
# @markdown We can further accelerate inference by setting the model to evaluation (inference)
# @markdown mode, which disables training-specific things like dropout layers.
gpu_device = torch.device("cuda")
model.to(gpu_device)
model.eval()


transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform


def estimate_depth(image):
    transformed_image = transform(image).to(gpu_device)

    with torch.no_grad():
        prediction = model(transformed_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    return output


def plot_comparison(image, output, save_path):
    figure = plt.figure(figsize=(32, 32))
    grid = figure.add_gridspec(1, 2)

    image_ax = figure.add_subplot(grid[0, 0])
    image_ax.axis("off")

    prediction_ax = figure.add_subplot(grid[0, 1])
    prediction_ax.axis("off")

    image_ax.imshow(image)
    prediction_ax.imshow(output, cmap="gray")
    # plt.show()
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    import os

    folder_path = "data/0024mod/"
    for image_path in os.listdir(folder_path):
        full_path = folder_path + image_path
        print(image_path)
        with open(full_path, "rb") as f:
            image_data = f.read()
            image = np.array(Image.open(io.BytesIO(image_data)))

        depthmap = estimate_depth(image)
        save_path = image_path.replace(".png", "_depth.png")
        depth_folder = "data/depth/"
        plt.imsave(depth_folder + save_path, depthmap, cmap="gray")
        plot_comparison(image, depthmap, depth_folder + image_path)
