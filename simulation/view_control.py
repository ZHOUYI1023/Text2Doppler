import numpy as np
from sklearn.decomposition import PCA

def adjust_origin(skel_hist, origin, delta_angle):
    """
    Adjust the origin coordinates based on the trajectory data and a delta angle.

    Parameters:
    skel_hist (np.ndarray): Skeleton history with trajectory data.
    origin (tuple): The original origin coordinates (x, y, z).
    delta_angle (float): The desired change in angle in degrees (positive for right, negative for left).

    Returns:
    tuple: New origin coordinates (x, y).
    """
    origin_x, origin_y, origin_z = origin

    # Extract x and y coordinates from the trajectory data
    x = skel_hist[0, 0, :]  # x-coordinates
    y = skel_hist[0, 2, :]  # y-coordinates

    # Translate the trajectory to the new origin
    translated_x = x - origin_x
    translated_y = y - origin_y
    translated_trajectory = np.column_stack((translated_x, translated_y))

    # Perform PCA to find the main direction
    pca = PCA(n_components=1)
    pca.fit(translated_trajectory)
    main_direction = pca.components_[0]
    angle = np.arctan2(main_direction[1], main_direction[0]) * 180 / np.pi

    # Calculate the desired angle and the new origin
    desired_angle = delta_angle + angle
    distance = np.sqrt((x[0] - origin_x) ** 2 + (y[0] - origin_y) ** 2)
    new_origin_x = x[0] - distance * np.sin(np.radians(angle - desired_angle))
    new_origin_y = y[0] - distance * np.cos(np.radians(angle - desired_angle))

    return new_origin_x, new_origin_y, origin_z
