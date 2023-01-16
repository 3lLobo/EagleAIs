from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
import matplotlib.pyplot as plt
import numpy as np
import torch


HUB_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"

feature_extractor = SegformerFeatureExtractor.from_pretrained(HUB_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(HUB_NAME)
image_processor = SegformerImageProcessor.from_pretrained(HUB_NAME)


def nvida_semantic(image: np.ndarray) -> np.ndarray:
    """Semantic segmentation with NVIDIA Segformer.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: segmented image
    """
    inputs = feature_extractor(images=image, return_tensors="pt")
    output = model(**inputs)
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


if __name__ == "__main__":
    import os
    from utils.image_utils import load_image, save_plot, semantic_overlay, save_image

    edit = "contrast"
    img_path = f"data/0024mod/0024{edit}.jpg"
    image = load_image(img_path)
    seg_img = nvida_semantic(image)
    save_folder = "res/NVIDIA/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    filename = img_path.split("/")[-1].replace(".jpg", "_seg.png")
    save_path = os.path.join(save_folder, filename)
    cmap = "jet"
    save_plot(seg_img, save_path, cmap=cmap)
    save_image(
        semantic_overlay(image, seg_img, alpha=0.9),
        save_path.replace(".png", "_overlay.png"),
    )
