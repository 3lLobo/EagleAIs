from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://user-images.githubusercontent.com/25290565/211220830-03fffc04-8020-4994-8be5-b2e75636a3e5.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

image_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
)
seg_img = image_processor.post_process_semantic_segmentation(outputs)
print(seg_img)

heat_img = seg_img[0].cpu().detach().numpy()
print(heat_img)


plt.imshow(heat_img, cmap="cool", interpolation="nearest")
plt.show()
plt.savefig("semanticSegmentationTest.png")
