import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation

def rotx(t):
    """3D Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -1.0 * (T_inv[:3, :3] @ T[:3, 3])
    return T_inv


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom


def convert_points_from_world(points, pose):
    hom_points = cart_to_hom(points[:, :3])
    converted_points = hom_points @ pose.T
    points[:, :3] = converted_points[:, :3]
    return points


def convert_boxes_from_n_to_vir(boxes, pose, drone=False):
    # vir pose
    # pose = self.get_virtual_pose(index, drone)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)

    r = Rotation.from_matrix(pose[:3, :3])
    ego2world_yaw = r.as_euler("xyz")[-1]
    boxes_global = boxes.copy()
    expand_centroids = np.concatenate([boxes[:, :3], np.ones((boxes.shape[0], 1))], axis=-1)
    centroids_global = np.dot(expand_centroids, pose.T)[:, :3]
    boxes_global[:, :3] = centroids_global
    boxes_global[:, 6] += ego2world_yaw
    return boxes_global


def convert_points_to_virtual(points, pose, drone=False):
    # print("==> ☁️ convert_points_to_virtual")
    # n --> virtual
    Ln_T_L0 = pose  # m3ed_utils.transform_inv(self.lidar_pose[index])
    r = Rotation.from_matrix(Ln_T_L0[:3, :3])
    Rn_T_R0_x, Rn_T_R0_y, _ = r.as_euler("xyz")
    R_x = rotx(Rn_T_R0_x)
    R_y = roty(Rn_T_R0_y)
    virtual_r_matrix = R_y @ R_x
    virtual_pose = np.eye(4)
    virtual_pose[:3, :3] = virtual_r_matrix

    if drone:
        position_n = Ln_T_L0[:3, 3]
        change_z = position_n[2]
        virtual_translate = np.eye(4)
        virtual_translate[2, 3] = change_z
        virtual_pose = virtual_translate @ virtual_pose

    virtual_points = convert_points_from_world(points, virtual_pose)
    return virtual_points, virtual_pose


if __name__ == "__main__":

    # ==== 示例参数 ====
    # bbox_center = np.array([3.95, -1.34, 0.30])
    # bbox_size = np.array([0.77, 0.73, 1.63])  # width, height, depth
    bbox = np.array([3.95, -1.34, 0.30, 0.77, 0.73, 1.63])

    # ==== 生成并保存 ====
    bbox_mesh = create_axis_aligned_bbox_with_cylindrical_edges(bbox_center, bbox_size, radius=0.02, color_rgb=(0, 180, 139))
    o3d.io.write_triangle_mesh("/home/rongl/code/3eed/converted_ply/bbox_axis_aligned_cylinder_edges.ply", bbox_mesh)

    print("✅ 保存为 'bbox_axis_aligned_cylinder_edges.ply'")
