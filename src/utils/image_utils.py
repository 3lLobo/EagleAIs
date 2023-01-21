from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def load_image(image_path: str) -> np.ndarray:
    """Load image from path.

    Args:
        image_path (str): Path to image

    Returns:
        np.ndarray: Image as numpy array
    """
    image = Image.open(image_path)
    return np.array(image)


def save_plot(
    image: np.ndarray,
    save_path: str,
    cmap: str = "nipy_spectral",
    figsize: Tuple[int, int] = None,
    id2label: Dict[int, str] = None,
) -> None:
    """Save image as plot.

    Args:
        image (np.ndarray): image to save
        save_path (str): path to save the image
        cmap (str): colormap
        figsize (Tuple[int, int]): figure size
        id2label (Dict[int, str]): mapping from id to label
    """
    if figsize is None:
        figsize = [image.shape[0] / 10, image.shape[1] / 10]
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    # TODO: Labels fir colors which occur in the image
    # Get all occuring labels
    # labels = np.unique(heat_img)
    # norm_labels = labels / labels.max()
    # # Get the colors of the values, according to the colormap used by imshow
    # colors = [plt.cm.nipy_spectral(norm_label) for norm_label in norm_labels]
    # # Create a patch (proxy artist) for every color
    # patches = [
    #     plt.plot(
    #         [],
    #         [],
    #         marker="s",
    #         ms=10,
    #         ls="",
    #         mec=None,
    #         color=colors[i],
    #         label=id2label[labels[i]],
    #     )
    #     for i in range(len(labels))
    # ]

    # legend = ax.legend(
    #     handles=patches,
    #     labels=[id2label[labels[i]] for i in range(len(labels))],
    #     bbox_to_anchor=(1.05, 1),
    #     loc=2,
    #     borderaxespad=0.0,
    #     title="Classes",
    #     fontsize="x-large",
    # )


def semantic_overlay(
    image: np.ndarray,
    heat_img: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "nipy_spectral",
    id2label: Dict[int, str] = None,
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image (np.ndarray): image to overlay on
        heat_img (np.ndarray): heatmap to overlay
        alpha (float): alpha value for overlay
        cmap (str): colormap
        id2label (Dict[int, str]): mapping from id to label

    Returns:
        np.ndarray: image with overlay
    """
    # Normalize heatmap
    heat_img = heat_img / heat_img.max()
    # Apply colormap
    heat_img = plt.cm.get_cmap(cmap)(heat_img)
    # Apply alpha
    # heat_img[:, :, 3] = alpha
    # Convert to numpy array
    heat_img = np.array(heat_img * 255, dtype=np.uint8)
    # Overlay heatmap on image
    overlay = cv2.addWeighted(image, 1, heat_img[:, :, :3], alpha, 0)
    return overlay


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save image to path.

    Args:
        image (np.ndarray): image to save
        save_path (str): path to save the image
    """
    Image.fromarray(image).save(save_path)


def plot_comparison(
    image: np.ndarray,
    output: np.ndarray,
    save_path: str = None,
) -> plt.figure:
    """Plot the original image and its
    transformation side by side.

    Args:
        image (np.ndarray): original image
        output (np.ndarray): transformed image
        save_path (str): path to save the image

    Returns:
        plt.figure: figure with comparison
    """
    figure = plt.figure(figsize=(32, 32))
    grid = figure.add_gridspec(1, 2)

    image_ax = figure.add_subplot(grid[0, 0])
    image_ax.axis("off")

    prediction_ax = figure.add_subplot(grid[0, 1])
    prediction_ax.axis("off")

    image_ax.imshow(image)
    prediction_ax.imshow(output, cmap="gray")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    return figure


# Calibration values for depth estimation to distance
PX1 = 230
PX2 = 169
DST1 = 100
DST2 = 200  # cm


def get_interpolation(
    pxvalue1: int, dist1: int, pxvalue2: int, dist2: int, x: int
) -> int:
    """Get distance for pixel value x.
    Given two points and their relative distance to the camera,
    interpolate the distance for point x.
    Pixel values are are 0 for the closest point and 255 for the farthest point, which is infinity.

    Args:
        pxvalue1 (int): pixel value of first point
        dist1 (int): distance to first point
        pxvalue2 (int): pixel value of second point
        dist2 (int): distance to second point
        x (int): pixel value to interpolate.

    Returns:
        int: distance to pixel value x
    """
    # Avoid division by zero
    if dist1 == dist2:
        return dist1
    # logarithmic interpolation
    result = np.exp(
        np.log(dist1)
        + (np.log(dist2) - np.log(dist1)) * (x - pxvalue1) / (pxvalue2 - pxvalue1)
    )
    return int(result)


def norm_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth image to range [0, 255].

    Args:
        depth (np.ndarray): depth image

    Returns:
        np.ndarray: normalized depth image
    """
    depth = depth - depth.min()
    depth = depth / depth.max()
    depth = depth * 255

    return depth.astype(np.uint8)


def get_distance(depth_map: np.ndarray, x: int, y: int) -> int:
    """Get distance from depth map.

    Args:
        depth_map (np.ndarray): normalized depth map
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        int: distance to point (x, y)
    """
    pxvalue = depth_map[y, x]
    return get_interpolation(PX1, DST1, PX2, DST2, pxvalue)


def get_area(
    depth_est: np.ndarray,
    x_max: int,
    y_max: int,
    x_min: int,
    y_min: int,
) -> int:
    """Get the area of circle.
    The values are the vertical max and min values of the bounding box.
    Area is in cm^2.

    Args:
        depth_est (np.ndarray): raw depth map
        x_max (int): x coordinate of max value
        y_max (int): y coordinate of max value
        x_min (int): x coordinate of min value
        y_min (int): y coordinate of min value

    Returns:
        int: area of circle
    """
    # Normalize depth map
    depth_norm = norm_depth(depth_est)
    # Get distance to max value
    dist_max = get_distance(depth_norm, x_max, y_max)
    # Get distance to min value
    dist_min = get_distance(depth_norm, x_min, y_min)
    # Get radius
    radius = (dist_max + dist_min) / 2
    # Get area of circle
    area = np.pi * radius**2
    return int(area)


def invert_depth(depth: np.ndarray) -> np.ndarray:
    """Invert depth image.

    Args:
        depth (np.ndarray): depth image

    Returns:
        np.ndarray: inverted depth image
    """
    return 255 - depth


if __name__ == "__main__":
    # Test the interpolation function

    x = 1
    pxvalue1 = 250
    dst1 = 100
    pxvalue2 = 230
    dst2 = 200
    dst_x = get_distance(pxvalue1, dst1, pxvalue2, dst2, x)
    print(dst_x)
