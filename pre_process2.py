import open3d as o3d
import numpy as np
import os
import cv2
import time

# todo 改为你自己的文件路径
path1 = '....../nutrition5k_dataset/imagery/realsense_overhead/'

num = 1

for i in os.listdir(path1):
    start_time = time.time()  # 记录开始时间
    print(num)
    num += 1
    if num > 10:
        break
    print(i)

    rgb_image_path = os.path.join(path1, i, "rgb.png")
    depth_image_path = os.path.join(path1, i, "depth_raw.png")

    if not os.path.exists(rgb_image_path):
        print(f"Folder {rgb_image_path} does not exist, skipping...")
        continue
    if not os.path.exists(depth_image_path):
        print(f"Folder {depth_image_path} does not exist, skipping...")
        continue

    # 读取 RGB 和深度图像
    rgb_image = o3d.io.read_image(rgb_image_path)  # RGB 图像
    depth_image = o3d.io.read_image(depth_image_path)  # 深度图像

    # 将 Open3D 图像转换为 numpy 数组
    rgb_image_np = np.asarray(rgb_image)
    depth_image_np = np.asarray(depth_image)

    # 目标尺寸
    target_width, target_height = 320, 448

    rgb_image_resized_np = rgb_image_np
    depth_image_resized_np = depth_image_np

    # 深度图噪声过滤：使用中值滤波
    depth_image_resized_np = cv2.medianBlur(depth_image_resized_np, 5)

    # 将深度图转换为浮动类型
    depth_image_resized_np = depth_image_resized_np.astype(np.float32)

    # 替换深度图中的 0 和大于 4000 的值为 NaN
    min_depth = 100  # 最小深度值（例如，500毫米）
    max_depth = 4000  # 最大深度值（例如，4000毫米）

    # 将小于最小值的深度值设置为最小值，大于最大值的深度值设置为最大值
    depth_image_resized_np[depth_image_resized_np < min_depth] = min_depth
    depth_image_resized_np[depth_image_resized_np > max_depth] = max_depth

    # 转换回 Open3D 图像格式
    rgb_image_resized = o3d.geometry.Image(rgb_image_resized_np)
    depth_image_resized = o3d.geometry.Image(depth_image_resized_np)

    # 创建 RGB-D 图像
    try:
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image_resized,
            depth_image_resized,
            depth_scale=2000.0,  # 深度单位：从毫米转换为米
            depth_trunc=4.0,  # 深度截断阈值（单位：米）
            convert_rgb_to_intensity=False  # 不将 RGB 转换为灰度图
        )

    except Exception as e:
        print(f"Error creating RGBD image: {e}")
        continue

    # 创建相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=target_width, height=target_height, fx=616.591, fy=616.765, cx=318.482, cy=241.167)

    # 使用 RGB-D 图像和相机内参创建点云
    try:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        continue

    # 输出生成的点云信息
    points = np.asarray(pcd.points)  # 点坐标 (N, 3)
    print(f"Number of points in the point cloud: {points.shape[0]}")

    end_time = time.time()  # 记录结束时间
    processing_time = end_time - start_time  # 计算处理时间
    fps = 1 / processing_time if processing_time > 0 else 0  # 计算 FPS
    print(f"Processing time: {processing_time:.4f} seconds, FPS: {fps:.2f}")

    # o3d.io.write_point_cloud(os.path.join(path1, i, "predict__point_cloud.ply"), pcd)
    o3d.io.write_point_cloud(os.path.join(path1, i, "processed_point_cloud.ply"), pcd)