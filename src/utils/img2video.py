from tqdm import tqdm
import ffmpeg


def img2video(img_dir: str, video_name: str):
    """Convert a directory of images to a video.

    Args:
        img_dir (str): Directory of images.
        video_name (str): Name of video.
    """
    (
        ffmpeg.input(img_dir + "/*.jpg", pattern_type="glob", framerate=25)
        .output(video_name)
        .run()
    )


if __name__ == "__main__":
    img_dir = "/home/wolf/worqspace/EagleEyez/runs/detect/predict"
    video_name = "./predict2.mp4"
    img2video(img_dir, video_name)
