import numpy as np
import cv2
import math

def combine_all_annotations(camera_pos, metahuman_transforms, metahuman_headpose_and_gaze, metahuman_keypoints):
    """
    Combine all raw annotations.
    """
    all_annotations = {
        "camera_pos": [camera_pos.x, camera_pos.y, camera_pos.z],
        "metahuman": {},
    }
    for actor_name, transform in metahuman_transforms.items():
        cleared_name = actor_name.split('_')[1]
        all_annotations["metahuman"][cleared_name] = {
            "translation": transform["translation"],
            "rotation": transform["rotation"],
            "headpose": metahuman_headpose_and_gaze[actor_name]["headpose"],
            "gaze": metahuman_headpose_and_gaze[actor_name]["gaze"],
            "keypoints": metahuman_keypoints[actor_name],
        }
    return all_annotations

def compute_intrinsics(width, height, fov_deg):
    fov_rad = np.radians(fov_deg)
    f = width / (2 * np.tan(fov_rad / 2))
    c_x = width / 2
    c_y = height / 2
    K = np.array([[f, 0, c_x],
                  [0, f, c_y],
                  [0, 0, 1]])
    return K

def project_pts_to_persp_image(points_3d, K):
    """
    Project 3D points onto a perspective image plane.

    Args:
        points_3d (numpy.ndarray): 3D points with shape (N, 3), each row is (X, Y, Z).
        K (numpy.ndarray): Camera intrinsic matrix with shape (3, 3).

    Returns:
        numpy.ndarray: 2D pixel coordinates with shape (N, 2), each row is (u, v).
    """
    points = points_3d / points_3d[:, 2:]  # (N, 3) -> (N, 3), normalize by Z
    points = K @ points.T                  # (3, 3) @ (3, N) -> (3, N)
    points_2d = points[:2].T               # (2, N) -> (N, 2)
    points_2d = np.round(points_2d).astype(np.int32)

    return points_2d

def project_pts_to_fisheye_image(points_3d, radius=1024, fov=180):
    """
    Project 3D points onto an equidistant fisheye image plane.

    Args:
        points_3d (numpy.ndarray): 3D points with shape (N, 3), each row is (X, Y, Z).
        radius (int): Fisheye image radius.
        fov (int): Fisheye field of view in degrees.

    Returns:
        numpy.ndarray: 2D pixel coordinates with shape (N, 2), each row is (u, v).
    """

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    theta = np.arctan(np.sqrt(x**2 + y**2) / z)  # calculate theta
    r = theta / (np.deg2rad(fov) / 2) * radius   # calculate r
    norm_x = x / np.sqrt(x**2 + y**2)
    norm_y = - y / np.sqrt(x**2 + y**2)
    u = r * norm_x + radius
    v = r * norm_y + radius
    points_2d = np.column_stack((u, v))
    points_2d = np.round(points_2d).astype(np.int32)
    return points_2d

def get_bounding_box(keypoints):
    points_2d = project_pts_to_fisheye_image(keypoints, radius=1024, fov=180)
    bounding_box = cv2.boundingRect(points_2d)
    bounding_box = (bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])
    # for point in points_2d:
        # cv2.circle(fisheye_image, tuple(point), 5, (0, 0, 255), -1)
    return bounding_box

def euler_to_matrix_ue(roll, pitch, yaw):
    # Convert degrees to radians.
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)  # Left-handed convention around the Z axis.
    
    # Rotation matrix around X axis (roll).
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), math.sin(roll_rad)],
        [0, -math.sin(roll_rad), math.cos(roll_rad)]
    ])
    
    # Rotation matrix around Y axis (pitch).
    Ry = np.array([
        [math.cos(pitch_rad), 0, -math.sin(pitch_rad)],
        [0, 1, 0],
        [math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])
    
    # Rotation matrix around Z axis (yaw).
    Rz = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    # Compose in X-Y-Z order.
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def rotate_keypoints(points, headpose, rotation, head_position):
    """Rotate keypoints at the origin to the headpose.

    Args:
        points (np.ndarray): keypoints when the head is at the origin (N x 3)
        headpose (list): headpose in euler angles (roll, pitch, yaw)
        rotation (list): body rotation in euler angles (roll, pitch, yaw)
        head_position (np.ndarray): head position (3,)

    Returns:
        np.ndarray: rotated keypoints (N x 3)
    """
    R = euler_to_matrix_ue(*headpose)
    R2 = euler_to_matrix_ue(-rotation[0], -rotation[1], -rotation[2])
    R3 = euler_to_matrix_ue(*rotation)
    points = (points - head_position) @ R2.T @ R.T @ R3.T + head_position
    return points

def get_simple_keypoints(points_2d, keypoint_labels, simple_keypoint_labels):
    simple_points = []
    for point, key in zip(points_2d, keypoint_labels):
        if key in simple_keypoint_labels:
            simple_points.append(point.tolist())
    return simple_points

def get_two_eye_center(points, points_2d, keypoint_labels):
    for point, point_2d, key in zip(points, points_2d, keypoint_labels):
        if key == "FACIAL_L_EyeCornerInner":
            left_eye = point
            left_eye_2d = point_2d
        if key == "FACIAL_R_EyeCornerInner":
            right_eye = point
            right_eye_2d = point_2d
    eye_center = (left_eye + right_eye) / 2
    eye_center = eye_center.astype(np.int32).tolist()
    eye_center_2d = (left_eye_2d + right_eye_2d) / 2
    eye_center_2d = eye_center_2d.astype(np.int32).tolist()
    return eye_center_2d, eye_center

def get_vectors(rotation, headpose, gaze):
    unit_vector = np.array([0, 1, 0])
    R_rotation = euler_to_matrix_ue(*rotation)
    body_vector = R_rotation @ unit_vector
    R_headpose = euler_to_matrix_ue(*headpose)
    head_vector = R_headpose @ body_vector
    R_gaze = euler_to_matrix_ue(0, gaze[0], -gaze[1])
    gaze_vector = R_gaze @ head_vector
    return body_vector, head_vector, gaze_vector

def visualize_vector(image, center, vector, color, radius=1024, fov=180):
    end_point = center + vector * 10
    points = np.array([center, end_point])
    points_2d = project_pts_to_fisheye_image(points, radius=radius, fov=fov)
    points_2d = points_2d.astype(np.int32)
    cv2.arrowedLine(image, tuple(points_2d[0]), tuple(points_2d[1]), color, 2)

def get_simple_annos(annos, simple_keypoint_labels):
    camera_pos = annos["camera_pos"]
    metahuman = annos["metahuman"]
    simple_annos = {}
    for actor_name, single_anno in metahuman.items():
        # print(actor_name)
        # print(single_anno["keypoints"].keys())
        rotation = single_anno["rotation"]
        headpose = single_anno["headpose"]
        gaze = single_anno["gaze"]
        points = np.array([np.array(kp) - np.array(camera_pos) for kp in single_anno["keypoints"].values()])
        points = rotate_keypoints(points, headpose, rotation, points[0])
        points_2d = project_pts_to_fisheye_image(points, radius=1024, fov=180)
        bbox = get_bounding_box(points[1:])
        simple_keypoints = get_simple_keypoints(points_2d, single_anno["keypoints"].keys(), simple_keypoint_labels)
        two_eye_center_2d, two_eye_center = get_two_eye_center(points, points_2d, single_anno["keypoints"].keys())
        body_vector, head_vector, gaze_vector = get_vectors(rotation, headpose, gaze)
        simple_annos[actor_name] = {"bbox": bbox, "two_eye_center": two_eye_center, "two_eye_center_2d": two_eye_center_2d, "keypoints": simple_keypoints,
                                    "body_vector": body_vector.tolist(), "head_vector": head_vector.tolist(), "gaze_vector": gaze_vector.tolist()}
    return simple_annos

def visualize_simple_annos(image, simple_annos):
    for actor_name, single_anno in simple_annos.items():
        bbox = single_anno["bbox"]
        simple_keypoints = single_anno["keypoints"]
        two_eye_center = single_anno["two_eye_center"]
        two_eye_center_2d = single_anno["two_eye_center_2d"]
        body_vector = np.array(single_anno["body_vector"])
        head_vector = np.array(single_anno["head_vector"])
        gaze_vector = np.array(single_anno["gaze_vector"])
        # Visualizations
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (0, 255, 255), (255, 0, 255), (127, 127, 255), (127, 255, 127)]
        for idx, point in enumerate(simple_keypoints):
            cv2.circle(image, tuple(point), 5, colors[idx], -1)
        cv2.circle(image, tuple(two_eye_center_2d), 5, (0, 255, 0), -1)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # visualize_vector(image, points[0], body_vector, (0, 255, 0))
        visualize_vector(image, two_eye_center, head_vector, (0, 255, 255))
        visualize_vector(image, two_eye_center, gaze_vector, (255, 0, 0))

def fisheye_2d_to_3d(pts_2d, d, radius=512, fov=180):
    """
    Convert a 2D point on an equidistant fisheye image to 3D coordinates.

    Args:
        pts_2d: 2D pixel coordinate (x, y).
        d: Distance from camera to the 3D point.
        radius: Fisheye image radius.
        fov: Fisheye field of view in degrees.

    Returns:
        np.ndarray: 3D coordinate (X, Y, Z).
    """
    x, y = pts_2d
    x_c = x - radius
    y_c = y - radius

    r = np.sqrt(x_c**2 + y_c**2)
    theta = r / radius * np.deg2rad(fov) / 2

    phi = np.arctan2(y_c, x_c)

    # Compute 3D coordinates.
    X = d * np.sin(theta) * np.cos(phi)
    Y = - d * np.sin(theta) * np.sin(phi)
    Z = d * np.cos(theta)

    return np.array([X, Y, Z])

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import os
    import json
    import tqdm
    from configs.mpsgaze360 import simple_keypoint_labels
    K = compute_intrinsics(1080, 1080, 120)

    export_dir = "MPSGaze360_20250228_804_frame"
    files = os.listdir(export_dir)
    names = [file.split("_")[0] for file in files if file.endswith('_0000.png')]
    names = list(set(names))

    for name in tqdm.tqdm(names):
        annos_path = f"{export_dir}/{name}_annotations.json"
        fisheye_path = f"{export_dir}/{name}.png"
        annos = json.load(open(annos_path, 'r'))
        fisheye_image = cv2.imread(fisheye_path)
        simple_annos = get_simple_annos(annos, simple_keypoint_labels)
        with open(f"{export_dir}/{name}_simple_annos.json", 'w') as f:
            json.dump(simple_annos, f, indent=4)
        visualize_simple_annos(fisheye_image, simple_annos)
        # cv2.imshow("image", cv2.resize(fisheye_image, (1024, 1024)))
        # cv2.waitKey(0)
        cv2.imwrite(f"{export_dir}/{name}_annotated.png", fisheye_image)