import numpy as np
import cv2

# Soure: https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image


def un_distort(image: np.ndarray) -> cv2.imwrite:
    """Undistort image using camera matrix and distortion coefficients
    .

    Args:
        image (np.ndarray): Image to undistort

    Returns:
        cv2.imwrite: Undistorted image
    """

    # src = cv2.imread(image_path)
    src = image
    width = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4, 1), np.float64)

    # TODO: add your coefficients here!
    k1 = -1.0e-5
    # negative to remove barrel distortion
    k2 = -0.000000001
    p1 = 0.0000
    p2 = -0.000

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2  # define center y
    cam[0, 0] = 10.0  # define focal length x
    cam[1, 1] = 10.00  # define focal length y

    # here the undistortion will be computed
    dst = cv2.undistort(src, cam, distCoeff)

    return dst
    # cv2.imwrite("un_dist.png", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    #     undistort(image_path)
    image_path = "/home/wolf/worqspace/EagleEyez/data/png/s1/0052.png"
    image = cv2.imread(image_path)
    un_distort(image)
    cv2.imwrite("un_dist.png", un_distort(image))
    print("Done")
