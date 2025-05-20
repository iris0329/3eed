import numpy as np
import open3d as o3d

import os

COLOR_RED = np.array([239, 99, 75]) / 255.0
COLOR_BLUE = np.array([99, 113, 250]) / 255.0
COLOR_GREEN = np.array([0, 180, 139]) / 255.0

def visualize_3eed_pointcloud_with_bbox(xyz, point_instance_label, save_path="./visualization.ply"):
    """
    Visualize 3EED point cloud and annotations, target points in red, other points in gray, saved as .ply file.

    Args:
        xyz: numpy array, shape (N, 3), point cloud coordinates
        point_instance_label: numpy array, shape (N,), points belonging to target object marked as 0, others as -1
        save_path: save path
    """
    # Initialize color array
    colors = np.zeros_like(xyz)

    # Target points in red [1, 0, 0], other points in gray [0.5, 0.5, 0.5]
    colors[point_instance_label == 0] = [1.0, 0.0, 0.0]  # red
    colors[point_instance_label == -1] = [0.5, 0.5, 0.5]  # gray

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save as .ply file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Point cloud saved to {save_path}")
    return


def save_as_ply(pc, intensity, save_path):
    # 使用距离渐变颜色而不是强度
    colors = compute_distance_colors(pc)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = o3d.io.write_point_cloud(save_path, pcd)
    assert success, f"Failed to save: {save_path}"
    print(f"✅ Saved: {save_path}")
    



def compute_distance_colors(points):
    """根据点距离的远近计算渐变颜色（蓝->绿->红）"""
    # 计算每个点到原点的距离
    distances = np.linalg.norm(points, axis=1)
    min_dist, max_dist = np.min(distances), np.max(distances)

    # 归一化距离到[0,1]
    normalized_dist = (distances - min_dist) / (max_dist - min_dist + 1e-5)

    # 创建颜色渐变：远距离(0.0)->蓝色，中距离(0.5)->绿色，近距离(1.0)->红色
    colors = np.zeros((points.shape[0], 3))

    # 分段线性插值
    mask1 = normalized_dist <= 0.3  # 蓝到绿的部分
    mask2 = normalized_dist > 0.3  # 绿到红的部分

    # 蓝->绿渐变
    t = normalized_dist[mask1] * 2  # 映射到[0,1]
    colors[mask1] = (1 - t)[:, np.newaxis] * COLOR_BLUE + t[:, np.newaxis] * COLOR_GREEN

    # 绿->红渐变
    t = (normalized_dist[mask2] - 0.5) * 2  # 映射到[0,1]
    colors[mask2] = (1 - t)[:, np.newaxis] * COLOR_GREEN + t[:, np.newaxis] * COLOR_RED

    return colors


def create_rotated_bbox_with_cylindrical_edges(bbox, radius=0.02, color_rgb=(0, 180, 139)):
    """
    Create a rotated bounding box with cylindrical edges.
    
    Args:
        bbox: numpy array of length 7, containing [x, y, z, width, height, depth, rotation_z]
              where rotation_z is the rotation angle around z-axis in radians
        radius: radius of the cylindrical edges
        color_rgb: RGB color tuple for the bbox
    """
    if bbox.shape[0] == 1 and len(bbox.shape) == 2:
        bbox = bbox.reshape(-1)

    assert len(bbox.shape) == 1, "bbox should be a 1D array"
    assert len(bbox) >= 7, "bbox should have at least 7 elements [x,y,z,w,h,d,rot_z]"

    center = bbox[:3]
    size = bbox[3:6]
    rotation_z = bbox[6]  # rotation around z-axis in radians

    w, h, d = size
    half = np.array([w, h, d]) / 2

    # 8 个顶点（局部坐标轴对齐）
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    vertices = signs * half

    # 创建绕z轴旋转的旋转矩阵
    cos_rot = np.cos(rotation_z)
    sin_rot = np.sin(rotation_z)
    rot_matrix = np.array([
        [cos_rot, -sin_rot, 0],
        [sin_rot, cos_rot, 0],
        [0, 0, 1]
    ])

    # 应用旋转
    vertices = np.dot(vertices, rot_matrix.T)
    # 平移到中心点
    vertices = vertices + center

    # 12 条边连接索引
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # 底面
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # 顶面
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # 垂直边
    ]

    # 颜色归一化
    color = np.array(color_rgb) / 255.0

    cylinders = []
    for start_idx, end_idx in edges:
        p1 = vertices[start_idx]
        p2 = vertices[end_idx]
        vec = p2 - p1
        height = np.linalg.norm(vec)
        if height < 1e-6:
            continue

        # 创建沿 z 轴的圆柱
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        cyl.paint_uniform_color(color)

        # 旋转圆柱到边的方向
        direction = vec / height
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if np.linalg.norm(axis) > 1e-6:
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
            cyl.rotate(rot, center=(0, 0, 0))

        # 平移到边中心
        cyl.translate((p1 + p2) / 2)
        cylinders.append(cyl)

    # 合并所有圆柱体为一个 Mesh
    bbox_mesh = cylinders[0]
    for c in cylinders[1:]:
        bbox_mesh += c

    return bbox_mesh

