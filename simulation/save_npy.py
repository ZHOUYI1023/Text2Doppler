import os
import numpy as np
from bvh import Bvh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualization import plot_ellipsoid, plot_line
from tqdm import tqdm

# plot type
plot_type = 'Line'
# Define your dataset and output folder paths
dataset_folder = '/home/yi/Desktop/momask-codes/generation/simple'
output_folder = '/home/yi/Desktop/momask-codes/signal_processing/generation/npy'
anim_folder = '/home/yi/Desktop/momask-codes/signal_processing/generation/anim'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(anim_folder):
    os.makedirs(anim_folder)

# Function to process your .bvh files
def process_bvh_file(file_path):
    # Add your processing code here
    anim = Bvh()
    anim.parse_file(file_path)
    positions_all, rotations_all = anim.all_frame_poses()
    positions_all = positions_all.transpose(1, 2, 0)
    ani = FuncAnimation(fig, update, fargs = (positions_all, ),frames=positions_all.shape[-1], blit=False)
    return positions_all, ani


def update(frame_ind, positions_all):
    ax.clear() 
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.view_init(elev=15, azim=140)
    plt.axis('off')
    
    positions = positions_all[:,:,frame_ind]
    positions[:, [-1, -2]] = positions[:, [-2, -1]]

    index_lists = {
        'left_leg': [0, 1, 2, 3, 4, 5],
        'right_leg': [0, 6, 7, 8, 9, 10],
        'middle': [0, 11, 12, 13, 14, 15, 16],
        'left_arm': [13, 17, 18, 19, 20, 21],
        'right_arm': [13, 22, 23, 24, 25, 26],
    }


    minor_radius_ratios = {'left_leg': 0.1, 'right_leg': 0.1, 'middle': 0.5, 'left_arm': 0.1, 'right_arm': 0.1}

    for part, indices in index_lists.items():
        for i in range(len(indices)-1):
            A = positions[indices[i]]
            B = positions[indices[i + 1]]
            if plot_type == 'Line':
            	plot_line(ax, A, B)
            elif plot_type == 'Ellipsoid':
                plot_ellipsoid(ax, A, B, minor_radius1_ratio=minor_radius_ratios[part], minor_radius2_ratio=minor_radius_ratios[part])
            	
# Iterate through each sub-folder and process .bvh files

for subdir in os.listdir(dataset_folder):
    subdir_path = os.path.join(dataset_folder, subdir)
    if os.path.isdir(subdir_path):
        output_subdir = os.path.join(output_folder, subdir)
        anim_subdir = os.path.join(anim_folder, subdir)
        print(output_subdir, anim_subdir)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        if not os.path.exists(anim_subdir):
            os.makedirs(anim_subdir)
        
        file_index = 0
        for file in os.listdir(subdir_path):
            if file.endswith('.bvh'):
                file_path = os.path.join(subdir_path, file)
                # initial figure
                fig = plt.figure(figsize=(3, 3))
                ax = fig.add_subplot(111, projection='3d')
                # Process the .bvh file
                processed_data, anim = process_bvh_file(file_path)
                # Save the processed data
                output_file = os.path.join(output_subdir, f'{file_index}.npy')
                anim_file = os.path.join(anim_subdir, f'{file_index}.gif')  
                np.save(output_file, processed_data)
                anim.save(anim_file, writer='imagemagick',fps=50 , dpi = 50)
                file_index += 1
                plt.close()
