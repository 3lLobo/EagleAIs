# Eagle Eyez

SDAIA hackathon project.

![0024contrast](https://user-images.githubusercontent.com/25290565/212572319-4b272e76-283a-4f6d-afc4-479b9ccd1ba5.jpg)

## How to run

Install dependencies:

```bash
poetry install
```

Load the data into `data/` folder.

Semantic segmentation:

```bash
poetry run python3 src/sem_seg_dpt.py
```

Depth estimation:

```bash
poetry run python3 src/depth_est_dpt.py
```

## Dataset

https://smartathon.hackerearth.com/

## Approach

1. Semantic Segmentation with NVIDIA VIT
2. Image depth estimation with Monodepth2 - This is how Tesla replaced Lidar!
3. Pothole detection with YOLOv7 - fine-tuned on manually annotated dataset.
4. Hold-up! How-to manual annotation pipeline?
5. Classic CV: Canny edge detection, Hough transform, Laplacian of Gaussian.
6. Video overlay with segmentation mask, depth indicator, pothole count and severity score.

## Remarks

Playing with image pre-processing techniques, reveals that reducing the exposure of the image improves the prediction quality of both the semantic segmentation and depth estimation models.
This might be due to the more drastic darkening of the road compared buildings, peripherals and sky or due to the notorious over-exposure of the dessert scenery. 

![0024contrast](https://user-images.githubusercontent.com/25290565/212501829-e3120acf-197f-4d74-86ec-99c5cfade208.jpg)

![0024exposure](https://user-images.githubusercontent.com/25290565/212501808-7c5d57a7-c97b-404b-9957-de41c7a2f5a8.jpg)


### CUDA

When working on WSL2, cuda is not readily available. Install it from the [official website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network). Then apply this [tweak](https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661) to your shells profile.

Congrats, your are squared away with CUDA!


## Resources

[pretrained Models](https://huggingface.co/)

[DPT paper](https://arxiv.org/pdf/2103.13413.pdf)