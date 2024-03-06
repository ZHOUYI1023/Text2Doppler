import numpy as np
import os
from visualization import save_spectrogram_image
from tqdm import tqdm

dataset_folder = '/home/yi/Desktop/momask-codes/signal_processing/spec_continuous'
#dataset_folder = '/media/yi/Backup Plus/ci4r'
output_folder = '/home/yi/Desktop/momask-codes/signal_processing/fig/sim_c'
#output_folder = '/home/yi/Desktop/momask-codes/signal_processing/fig/ci4r'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
train_spec_data = os.path.join(dataset_folder, 'train')
test_spec_data = os.path.join(dataset_folder, 'test')    

for subdir in os.listdir(train_spec_data):
    subdir_spec_path = os.path.join(train_spec_data, subdir)
    if os.path.isdir(subdir_spec_path):
        # Create train and test subdirectories
        train_fig_subdir = os.path.join(output_folder, 'train', subdir)
        if not os.path.exists(train_fig_subdir):
            os.makedirs(train_fig_subdir)

            
        spec_files = [file for file in os.listdir(subdir_spec_path) if file.endswith('.npy')]
       
        for file_index, file in tqdm(enumerate(spec_files), total=len(spec_files), desc=f"Processing training set for {subdir}"):
            file_path = os.path.join(subdir_spec_path, file)
            save_path = os.path.join(train_fig_subdir, file[:-4]+'.png')
            result = np.load(file_path, allow_pickle=True)
            sx2 = result.item().get('spec')
            save_spectrogram_image(sx2, spec_length = 1600, dynamic_range = 60, image_path=save_path)
# 1600 60 / 400 30
            
            
for subdir in os.listdir(test_spec_data):
    subdir_spec_path = os.path.join(test_spec_data, subdir)
    if os.path.isdir(subdir_spec_path):
        # Create train and test subdirectories
        test_fig_subdir = os.path.join(output_folder, 'test', subdir)
        if not os.path.exists(test_fig_subdir):
            os.makedirs(test_fig_subdir)
            
        spec_files = [file for file in os.listdir(subdir_spec_path) if file.endswith('.npy')]
       
        for file_index, file in tqdm(enumerate(spec_files), total=len(spec_files), desc=f"Processing test set for {subdir}"):
            file_path = os.path.join(subdir_spec_path, file)
            save_path = os.path.join(test_fig_subdir, file[:-4]+'.png')
            result = np.load(file_path, allow_pickle=True)
            sx2 = result.item().get('spec')
            save_spectrogram_image(sx2, spec_length = 4000, dynamic_range = 60, image_path=save_path)
 


