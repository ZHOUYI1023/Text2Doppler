import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D


def save_spectrogram_image(sx2, spec_length = 1600, dynamic_range = 60, image_path='spectrogram.png'):
    """
    Crop or pad the sx2 data to a fixed length of spec_length and save the data as an image.
    The image size is fixed at [224, 224].
    """
    # Crop or pad sx2 data to a fixed size 
    current_size = sx2.shape[1]
    if current_size < spec_length:
        # Pad
        pad_width = spec_length - current_size
        min_val = np.min(sx2)
        sx2_padded = np.pad(sx2, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_val)
    elif current_size > spec_length:
        start_index = (current_size - spec_length) // 2
        sx2_padded = sx2[:, start_index:start_index + spec_length]
    else:
        sx2_padded = sx2

    spec_max = np.log10(np.abs(sx2_padded / np.max(sx2_padded))).max()
    # Visualization and saving
    plt.figure()  
    plt.imshow(20 * np.log10(np.abs(sx2_padded / np.max(sx2_padded))), aspect='auto', origin='lower')
    plt.clim(spec_max - dynamic_range, spec_max)
    plt.set_cmap('jet')
    plt.axis('off')  # No axes for an image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding around the image
    # Save the image
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory
    
    
def plot_ellipsoid(ax, A, B, minor_radius1_ratio=0.1, minor_radius2_ratio=0.1):
    """
    Plots an ellipsoid given two endpoints A and B of its major axis.

    Parameters:
    A (np.array): The first endpoint of the major axis.
    B (np.array): The second endpoint of the major axis.
    minor_radius1_ratio (float): Ratio of the major radius to use for the first minor radius.
    minor_radius2_ratio (float): Ratio of the major radius to use for the second minor radius.
    """
    # Calculate center and major radius
    center = (A + B) / 2
    major_radius = np.linalg.norm(B - A) / 2

    # Define minor radii based on ratios
    minor_radius1 = major_radius * minor_radius1_ratio
    minor_radius2 = major_radius * minor_radius2_ratio

    # Parametric equations for the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Calculate the ellipsoid surface points
    direction = (B - A) / np.linalg.norm(B - A)
    rotation_matrix = np.array([[direction[0], -direction[1], direction[2]],
                                [direction[1], direction[0], direction[2]],
                                [direction[2], -direction[2], direction[0]]])

    x = major_radius * np.outer(np.cos(u), np.sin(v))
    y = minor_radius1 * np.outer(np.sin(u), np.sin(v))
    z = minor_radius2 * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.dot(rotation_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]))
    x, y, z = xyz.reshape((3, x.shape[0], x.shape[1]))

    # Position the ellipsoid at the center
    x += center[0]
    y += center[1]
    z += center[2]

    # Plotting the ellipsoid
    ax.plot_surface(x, y, z, color='b', alpha=0.5)

    # Plotting endpoints and center for reference
    ax.scatter(*A, color='r')
    ax.scatter(*B, color='r')
    #ax.scatter(*center, color='g')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    #ax.set_title('Ellipsoid with Endpoints on Major Axis')
    
    
def plot_line(ax, A, B):
    """
    Plots a line given two endpoints A and B.

    Parameters:
    A (np.array): The first endpoint of the major axis.
    B (np.array): The second endpoint of the major axis.
 
    """
  
    # Plotting the ellipsoid
    ax.plot([A[0],B[0]],[A[1],B[1]],[A[2],B[2]], color='b')

 
