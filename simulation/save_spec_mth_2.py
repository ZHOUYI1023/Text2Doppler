import argparse
import numpy as np
import os
import random
from view_control import adjust_origin
from simulation import process_spectrogram
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_file(args):
    file_path, output_path, origin, delta_angle, rcs_flag = args
    skel_hist = np.load(file_path)
    new_origin = adjust_origin(skel_hist, origin, delta_angle)
    sx2, timeAxis, freqAxis = process_spectrogram(skel_hist, new_origin, rcs_flag)
    result = {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
    np.save(output_path, result)

def process_subdir(args):
    subdir_path, output_path, rcs_flag = args
    if not os.path.isdir(subdir_path):
        return

    skel_files = [file for file in os.listdir(subdir_path) if file.endswith('.npy')]
    random.shuffle(skel_files)  # Shuffle for randomness

    tasks = []
    for file_index, file in enumerate(skel_files):
        file_path = os.path.join(subdir_path, file)
        output_file = os.path.join(output_path, f'{file_index}.npy')
        origin = (0, -3, 0.5)  # Define the original origin
        delta_angle = random.uniform(-15, 15)  # Generate a random delta angle
        tasks.append((file_path, output_file, origin, delta_angle, rcs_flag))

    return tasks

def main(dataset_folder, output_folder, num_workers, rcs_flag):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    subdirs = [os.path.join(dataset_folder, subdir) for subdir in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, subdir))]
    tasks = []
    for subdir in subdirs:
        train_output_path = os.path.join(output_folder, 'train', os.path.basename(subdir))
        test_output_path = os.path.join(output_folder, 'test', os.path.basename(subdir))
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)
        tasks.extend(process_subdir((subdir, train_output_path, rcs_flag)))
        #tasks.extend(process_subdir((subdir, test_output_path, rcs_flag)))
      

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, tasks), total=len(tasks)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process spectrogram generation with multiprocessing.')
    parser.add_argument('--rcs_flag', type=bool, default=True, help='Consider RCS or not.')
    parser.add_argument('--dataset_folder', type=str, default='/home/yi/Desktop/momask-codes/signal_processing/generation/npy_filtered', help='Path to the dataset folder.')
    parser.add_argument('--output_folder', type=str, default='/home/yi/Desktop/momask-codes/signal_processing/spec_2', help='Path to the output folder.')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of worker processes to use.')
    args = parser.parse_args()

    main(args.dataset_folder, args.output_folder, args.workers, args.rcs_flag)
