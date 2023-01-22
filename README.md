# Eagle AI
![banner](https://user-images.githubusercontent.com/25290565/213936713-b867c032-54f2-4316-a568-5a9490252420.png)


SDAIA hackathon project.

[Demo Video](https://youtu.be/WpQN7lC9WMw)

[Demo notebook]()

## Project Description

Eagle AIs is modular, scalable  and a cutting-edge solution for the automated detection, localization and mitigation of street damage, yielding in a modern road network with automated sustenance workflow.

Engineered and prototyped by a team of young PhD candidates, the presented solution synergizes the power of Deep Learning, Cloud-computing and augmented-reality.

We solve the challenge of detection, localization and severity estimation of road-damage, provide exact geo-coordinates and 3D-reconstruction and eliminate the need for patrolling and specialized vehicles.

Data can be provided through a mobile-phone camera by ordinary citizens. While the geo-coordinates and time-stamp give an exact reference, the images get processed in batches on a cloud-environment. From a single image, we are able to detect different kinds of street damage, estimate the size and severity; moreover generate a complete 3D-reconstruction of the scene for on-site inspection with augmented-reality. 
Besides the standard detection of potholes, we decided to increase fidelity by expanding the detection capabilities with 2 additional classes, longitudinal road-damage and sand-accumulation.
While potholes proposing the most prominent risk to vehicle and driver require immediate attention, smaller and often expanded road damage such as cracks and water puddles are categorized with lower priority, yet shall be targets of pre-emptive care.
Finally we decided to include sand-accumulation as a separate class since it hinders visibility and could potentially cover severe damage, therefore shall be depper investigated for appropriate action.
The severity of the damage is estimated by combining a Laplacian-filter, a classical and reputable computer vision technique with 3D reconstruction for depth-estimation.
In combination with the GPS coordinates from the phone we provide a modular severity heatmap in a panoptic birds-eye view, combinable with popular map providers such as GoogleMaps. This is intended to optimize the coordination and logistics of repair-missions.
Project Eagle AIs is scalable, low-cost and highly performant.

## Screenshots


![obj_detection](https://user-images.githubusercontent.com/25290565/213936717-3d35064d-551f-477a-be47-4d6f5fd41d63.png)
![segmented_overlay](https://user-images.githubusercontent.com/25290565/213936722-4b510bb2-0983-4751-b9ac-ac451b297f67.png)
![augmented_reality](https://user-images.githubusercontent.com/25290565/213936726-57e1a11a-c45c-4e69-a80e-ae3f0d651d5e.png)

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