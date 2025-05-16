import numpy as np


def transform_to_front_view(points, boxes, source_view):

    rotation_angles = {
        "F": 0,
        "FL": -45,
        "FR": 45,
        "SL": -90,
        "SR": 90,
    }

    translation_vectors = {
        "F": [1.5, 0.0, 1.2],
        "FL": [1.2, 0.7, 1.2],
        "FR": [1.2, -0.7, 1.2],
        "SL": [0.0, 0.9, 1.2],
        "SR": [0.0, -0.9, 1.2],
    }

    angle_rad = np.radians(rotation_angles[source_view])
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotation_matrix = np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])

    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_boxes = boxes.copy()
    expand_centroids = rotated_boxes[:, :3]
    centroids = np.dot(expand_centroids, rotation_matrix.T)
    rotated_boxes[:, :3] = centroids
    rotated_boxes[:, 6] += angle_rad

    return rotated_points, rotated_boxes
