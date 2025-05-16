import numpy as np

def rotate_points(points, center, angle):
    # Rotate point cloud around z-axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])
    centered_points = points - center
    rotated_points = np.dot(centered_points, rotation_matrix.T)
    return rotated_points + center


def get_points_in_bbox(points, bbox_center, bbox_size, heading_angle):
    # Convert bbox_center and bbox_size to NumPy arrays
    bbox_center = np.array(bbox_center)
    bbox_size = np.array(bbox_size)

    # Rotate point cloud around bbox center in opposite direction
    # TODO consider healing angle, i.e., rotate bbox
    rotated_points = rotate_points(points, bbox_center, -heading_angle)

    # Calculate bbox minimum and maximum coordinates
    bbox_min = bbox_center - bbox_size / 2
    bbox_max = bbox_center + bbox_size / 2

    # Get points belonging to bbox
    mask = np.all((rotated_points >= bbox_min) & (rotated_points <= bbox_max), axis=1)
    # points_in_bbox = points[mask]
    return mask