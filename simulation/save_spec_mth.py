import numpy as np
import os
import random
from concurrent.futures import ProcessPoolExecutor
from view_control import adjust_origin
from simulation import process_spectrogram

def process_file(args):
    file, subdir_path, output_subdir, is_train = args
    file_path = os.path.join(subdir_path, file)
    skel_hist = np.load(file_path)
    if is_train:
        delta_angle = random.uniform(-15, 15)  
    else:
        delta_angle = 0
    origin = (0, -3, 0.5)
    new_origin = adjust_origin(skel_hist, origin, delta_angle) if is_train else origin
    sx2, timeAxis, freqAxis = process_spectrogram(skel_hist, new_origin, rcs_flag=True)
    result = {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
    output_file = os.path.join(output_subdir, f'{file}.npy')
    np.save(output_file, result)


dataset_folder = '/home/yi/Desktop/momask-codes/signal_processing/generation'
output_folder = '/home/yi/Desktop/momask-codes/signal_processing/spec'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir in os.listdir(dataset_folder):
    subdir_path = os.path.join(dataset_folder, subdir)
    if os.path.isdir(subdir_path):
        train_subdir = os.path.join(output_folder, 'train', subdir)
        test_subdir = os.path.join(output_folder, 'test', subdir)
        for subfolder in [train_subdir, test_subdir]:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

        skel_files = [file for file in os.listdir(subdir_path) if file.endswith('.npy')]
        random.shuffle(skel_files)

        split_index = int(0.8 * len(skel_files))
        train_files = skel_files[:split_index]
        test_files = skel_files[split_index:]

        with ProcessPoolExecutor() as executor:
            executor.map(process_file, [(file, subdir_path, train_subdir, True) for file in train_files])
            executor.map(process_file, [(file, subdir_path, test_subdir, False) for file in test_files])
