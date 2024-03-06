import numpy as np
import os
import random
from view_control import adjust_origin
from simulation import process_spectrogram
from tqdm import tqdm

dataset_folder = '/home/yi/Desktop/momask-codes/signal_processing/generation/npy_filtered'
output_folder = '/home/yi/Desktop/momask-codes/signal_processing/spec_filtered'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for subdir in os.listdir(dataset_folder):
    subdir_path = os.path.join(dataset_folder, subdir)
    if os.path.isdir(subdir_path):
        # Create train and test subdirectories
        train_subdir = os.path.join(output_folder, 'train', subdir)
        test_subdir = os.path.join(output_folder, 'test', subdir)
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
        if not os.path.exists(test_subdir):
            os.makedirs(test_subdir)
            
        skel_files = [file for file in os.listdir(subdir_path) if file.endswith('.npy')]
        random.shuffle(skel_files)  # Shuffle the list for randomness
        
        # Split files into 80% train and 20% test
        split_index = int(0.8 * len(skel_files))
        train_files = skel_files[:split_index]
        test_files = skel_files[split_index:int(len(skel_files))]
        
        # Process and save files for train set
        for file in tqdm(train_files, total=len(train_files), desc=f"Processing training set for {subdir}"):
            file_path = os.path.join(subdir_path, file)
            skel_hist = np.load(file_path)
            # Use original file name with a prefix or suffix to avoid overwriting
            output_file = os.path.join(train_subdir, f'train_{file}') 
            # Define the original origin
            origin = (0, -3, 0.5)
            # Generate a random delta angle between -15 and 15 degrees
            delta_angle = random.uniform(-15, 15)
            new_origin = adjust_origin(skel_hist, origin, delta_angle)
            sx2, timeAxis, freqAxis = process_spectrogram(skel_hist, new_origin, rcs_flag = True)
            result = {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
            np.save(output_file, result)

        # Process and save files for test set
        for file in tqdm(test_files, total=len(test_files), desc=f"Processing test set for {subdir}"):
            file_path = os.path.join(subdir_path, file)
            skel_hist = np.load(file_path)
            # Use original file name with a prefix or suffix to avoid overwriting
            output_file = os.path.join(test_subdir, f'test_{file}') 
            # Define the radar origin
            origin = (0, -3, 0.5)
            sx2, timeAxis, freqAxis = process_spectrogram(skel_hist, origin, rcs_flag = True)
            result = {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
            np.save(output_file, result)
