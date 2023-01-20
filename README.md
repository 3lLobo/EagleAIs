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

1. Semantic Segmentation with NVIDIA VIT and [DPTlarge-ade(https://huggingface.co/Intel/dpt-large-ade)]. ✅
2. Image depth estimation with MiDas and [DTPlarge](https://huggingface.co/Intel/dpt-large) - Tesla was the first one to replaced Lidar with monocular depth estimation! ✅
3. LabelStudio for manual annotation. ✅
4. Pothole detection with YOLOv8 - fine-tuned on manually annotated dataset.  ✅
5. Classic CV: Canny edge detection, Hough transform, Laplacian of Gaussian.
6. Fancy Video:
   1. Overlay with segmentation. ✅
   2. Depth indicator with distance measure.
   3. Increasing pothole count.
   4. Pothole bounding box and severity score.
   5. Street damage barometer.
7. Demo Video.
8. Wirte-up in paper style.
9. Submittt!


## Label Studio

Manual labeling was done in [Label Studio](https://labelstud.io/). The 'docker-compose.yml' file  is adopted from the [labelImg repo](https://github.com/heartexlabs/label-studio).
To start label studio, create the required folderstructure with `g+wr` permissions:
- data
  - import
  - export
  - media

Then run:
```bash
cd labelImg && docker-compose up
```

Open the browser at `localhost:8080`, set up the project and import the images for labeling.

~30 images were annotated for 3 classes:
 - pothole
 - streetdamage
 - sand on road

![image](https://user-images.githubusercontent.com/25290565/212654523-63fdfbd9-76be-4f1f-9fe1-3bb3316d56eb.png)

The annotations are exported in COCO format.


## Fine-tuning YOLOv8

Follow these [instructions](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) to save your labeled data and train a custom model.

We use the [YOLOv8](https://github.com/ultralytics/ultralytics) model.


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