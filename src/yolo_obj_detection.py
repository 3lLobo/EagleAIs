from ultralytics import YOLO
import ultralytics
from typing import List


def yolo_fine_tune(data_path: str, epochs: int):
    """Fine tune the YOLO model on a custom dataset.

    Args:
        data_path (str): Path to dataset in yolo format.
        epochs (int): Epochs to train for.
    """
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    results_train = model.train(data=data_path, epochs=epochs)  # train the model
    results_val = model.val()  # evaluate model performance on the validation set
    success = model.export(format="onnx")  # export the model to ONNX format


def yolo_inference(
    image_path: List[str],
    model_path: str,
    save: bool = True,
) -> List[str]:
    """Inference on a list of images.

    Args:
        image_path (List[str]): List of image paths.
        model_path (str): Path to model.

    Returns:
        List[str]: Result object
    """
    model = YOLO(model_path)
    results = model.predict(image_path, save=save)
    return results
