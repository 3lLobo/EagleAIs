import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os

from depth_est_dpt import dpt_depth
from utils.image_utils import norm_depth, invert_depth
from canny_edge import canny_edge
from utils.distorted_img import un_distort

device = o3d.core.Device("cuda:0")
dtype = o3d.core.float32


def get_rgbd(
    depth: np.ndarray, image_path: str, canny: bool = False, gpu: bool = False
) -> o3d.geometry.PointCloud:
    """Get a point cloud from a depth image.

    Args:
        depth (np.ndarray): Depth image.
        image_path (str): Path to image.
        canny (bool): Whether to use canny edge detection.
        gpu (bool): Whether to use gpu.

    Returns:
        o3d.geometry.PointCloud: Point cloud.
    """
    #  get the camera intrinsics
    C3dImage = o3d.geometry.Image
    if gpu:
        C3dImage = o3d.t.geometry.Image

    depth = np.log(depth + np.abs(depth.min()) + 1)
    # depth = invert_depth(norm_depth(depth))
    # depth = depth / 255
    # depth = np.power(depth, 5) * np.ones_like(depth) * 255
    # depth_norm = undistort(depth)
    depth_n = norm_depth(depth)
    depth_raw = C3dImage(depth_n.astype(np.float32))

    # Color image
    color_raw = cv2.imread(image_path)
    if canny:
        color_raw = canny_edge(np.array(color_raw), 50, 130)
    # color_raw = undistort(color_raw)
    color_raw = C3dImage(color_raw)

    RGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth
    if gpu:
        RGBDImage = o3d.t.geometry.RGBDImage

    rgbd_image = RGBDImage(
        color_raw,
        depth_raw,
        # convert_rgb_to_intensity=False,
        # depth_scale=1.0,
        # depth_trunc=3.0,
    )
    # plt.subplot(1, 2, 1)
    # plt.title("Original image")
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title("Depth image")
    # plt.imshow(rgbd_image.depth)
    # plt.savefig("depth_compare.png")

    return rgbd_image


def get_pcd_gpu(
    rgbd_image: o3d.t.geometry.RGBDImage, image_path: str
) -> o3d.geometry.PointCloud:
    """
    Get a point cloud from a depth image.

    Args:
        rgbd_image (o3d.t.geometry.RGBDImage): RGBD image.
        image_path (str): Path to image.

        Returns:
            o3d.geometry.PointCloud: Point cloud.
    """
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.cuda.pybind.core.Tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        with_normals=True,
    )
    pcd.cuda(0)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1.5, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])

    pcd.remove_non_finite_points()

    # o3d.visualization.draw_geometries_with_vertex_selection(
    #     [pcd.to_legacy(), pcd.to_legacy()]
    # )
    pcd.scale(10, center=pcd.get_center())
    return pcd


def get_pcd_cpu(
    rgbd_image: o3d.t.geometry.RGBDImage, image_path: str
) -> o3d.geometry.PointCloud:
    """
    Get a point cloud from a depth image.

    Args:
        rgbd_image (o3d.t.geometry.RGBDImage): RGBD image.
        image_path (str): Path to image.

        Returns:
            o3d.geometry.PointCloud: Point cloud.
    """
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        ),
    )
    # depth_max = np.max(np.asanyarray(pcd.points)[:, 2])
    # depth_min = np.min(np.asanyarray(pcd.points)[:, 2])
    # print(depth_max)
    # Filter out the horizon points
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # pcd = pcd.crop(
    #     o3d.geometry.AxisAlignedBoundingBox(
    #         min_bound=[-5, -5, depth_min + np.abs(depth_min) * 0.05],
    #         max_bound=[5, 5, depth_max * 0.89]
    #         # min_bound=[-5, -5, depth_min + 0.0001],
    #         # max_bound=[5, 5, depth_max - 0.00001],
    #     )
    # )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1.5, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_plotly([pcd])

    pcd.remove_non_finite_points()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    # o3d.visualization.draw_geometries_with_vertex_selection(
    #     [pcd]
    # )
    pcd.scale(10, center=pcd.get_center())
    return pcd


def show_pcd(
    pcds: o3d.geometry.PointCloud, background: np.ndarray = None, gpu: bool = False
):
    """Show the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.
        background (np.ndarray, optional): Background image. Defaults to None.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcds:
        if gpu:
            pcd = pcd.to_legacy()
        vis.add_geometry(pcd)
    if background is not None:
        vis.add_geometry(o3d.geometry.Image(background))
    vis.run()
    vis.destroy_window()


def move_pcd(
    pcd: o3d.geometry.PointCloud, test_data_path: str, camera_trajectory_path: str
):
    """Move the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.
        test_data_path (str): Path to test data.
        camera_trajectory_path (str): Path to camera trajectory.
    """
    custom_draw_geometry_with_camera_trajectory = o3d.visualization.Visualizer()
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = (
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    )
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    image_path = os.path.join(test_data_path, "image")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(test_data_path, "depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(
                os.path.join(depth_path, "{:05d}.png".format(glb.index)),
                np.asarray(depth),
                dpi=1,
            )
            plt.imsave(
                os.path.join(image_path, "{:05d}.png".format(glb.index)),
                np.asarray(image),
                dpi=1,
            )
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True
            )
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(
                None
            )
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(), target_temp.to_legacy()],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )
    return source_temp, target_temp


# def draw_ipc():
def draw_ipc(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_rgbd: o3d.geometry.RGBDImage,
    target_rgbd: o3d.geometry.RGBDImage,
):
    """Draw the IPC.

    Args:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.
        source_rgbd (o3d.geometry.RGBDImage): Source RGBD image.
        target_rgbd (o3d.geometry.RGBDImage): Target RGBD image.
    """
    transformation = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        source_rgbd,
        target_rgbd,
        o3d.cuda.pybind.core.Tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
    )
    # For Colored-ICP `colors` attribute must be of the same dtype as `positions` and `normals` attribute.
    source.point["colors"] = source.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
    target.point["colors"] = target.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

    # # Colored-ICP
    # reg_p2p = o3d.pipelines.registration.registration_colored_icp(
    #     source,
    #     target,
    #     0.05,
    #     # transformation.transformation,
    #     0.03,
    # )

    draw_registration_result(source, target, transformation.transformation)


def draw_odometry_cpu(
    target_pcd: o3d.geometry.PointCloud,
    source_rgbd_image: o3d.geometry.RGBDImage,
    target_rgbd_image: o3d.geometry.RGBDImage,
    save_name: str = None,
):
    """Draw the odometry.

    Args:
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        source_rgbd (o3d.geometry.RGBDImage): Source RGBD image.
        target_rgbd (o3d.geometry.RGBDImage): Target RGBD image.
        save_name (str, optional): Save name. Defaults to None.
    """
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )

    [
        success_color_term,
        trans_color_term,
        info,
    ] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        pinhole_camera_intrinsic,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
        option,
    )
    [
        success_hybrid_term,
        trans_hybrid_term,
        info,
    ] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        pinhole_camera_intrinsic,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic
        )
        source_pcd_color_term.transform(trans_color_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term])
        if save_name:
            o3d.io.write_point_cloud(
                f"./o3d/pcd_colorterm_{save_name}.ply",
                source_pcd_color_term,
                print_progress=True,
            )

    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic
        )
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
        if save_name:
            o3d.io.write_point_cloud(
                f"./o3d/pcd_hybridterm_{save_name}.ply",
                source_pcd_hybrid_term,
                print_progress=True,
            )
    if save_name:
        o3d.io.write_point_cloud(
            f"./o3d/pcd_target_{save_name}.ply",
            target_pcd,
            print_progress=True,
        )


if __name__ == "__main__":

    # image_pat h = "/home/wolf/worqspace/EagleEyez/runs/detect/predict2/image109.jpg"
    image_path = "/home/wolf/worqspace/EagleEyez/data/png/s1/0052.png"
    image_path2 = "/home/wolf/worqspace/EagleEyez/data/png/s1/0053.png"

    depth = dpt_depth(plt.imread(image_path))
    depth2 = dpt_depth(plt.imread(image_path2))

    # #  GPU
    # rgbd_image = get_rgbd(depth, image_path, gpu=True)
    # rgbd_image2 = get_rgbd(depth2, image_path2, gpu=True)
    # pcd = get_pcd_gpu(rgbd_image, image_path)
    # pcd2 = get_pcd_gpu(rgbd_image2, image_path2)

    # draw_ipc(pcd, pcd2, rgbd_image, rgbd_image2)

    # Now CPU
    rgbd_image = get_rgbd(depth, image_path, gpu=False)
    rgbd_image2 = get_rgbd(depth2, image_path2, gpu=False)
    pcd = get_pcd_cpu(rgbd_image, image_path)
    pcd2 = get_pcd_cpu(rgbd_image2, image_path2)

    draw_odometry_cpu(pcd, rgbd_image, rgbd_image2, save_name="0052_0053")
