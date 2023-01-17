from transformers import (
    DPTFeatureExtractor,
    DPTForSemanticSegmentation,
    DPTImageProcessor,
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import torch


feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large-ade")
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dpt_semantic(image: np.ndarray) -> np.ndarray:
    """Semantic segmentation with DPT.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: segmented image
    """
    with torch.cuda.device(0):
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        gpumodel = model.to(device)
        output = gpumodel(**inputs)
        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            # pp_image[0].unsqueeze(0).unsqueeze(0),
            output.logits,
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        output.logits = prediction
        pp_image = image_processor.post_process_semantic_segmentation(output)
    seg_img = pp_image[0].cpu().detach().numpy()

    return seg_img


def dpt_get_labels() -> Dict[int, str]:
    """Get semantic labels for DPT.

    Returns:
        Dict[int, str]: labels
    """
    id2label = model.config.id2label
    return id2label


if __name__ == "__main__":
    import os
    from utils.image_utils import load_image, save_plot, semantic_overlay, save_image

    edit = "contrast"
    img_path = f"data/0024mod/0024{edit}.jpg"
    image = load_image(img_path)
    seg_img = dpt_semantic(image)
    save_folder = "res/DPT/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    filename = img_path.split("/")[-1].replace(".jpg", "_seg.png")
    save_path = os.path.join(save_folder, filename)
    cmap = "jet"
    save_plot(seg_img, save_path, cmap=cmap)
    save_plot(seg_img, save_path, cmap=cmap)
    save_image(
        semantic_overlay(image, seg_img, alpha=0.9),
        save_path.replace(".png", "_overlay.png"),
    )
