import numpy as np
import scipy.constants
from scipy.io import loadmat
import os
import random
from view_control import adjust_origin
from simulation import load_config, generate_spectrogram
from tqdm import tqdm

dataset_folder = '/media/yi/Lenovo/gesture_datasets/CI4R_Alabama/new'
output_folder = '/media/yi/Backup Plus/ci4r'

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
         
        skel_files = [file for file in os.listdir(subdir_path) if file.endswith('.mat')]
        random.shuffle(skel_files)  # Shuffle the list for randomness
        
        # Split files into 80% train and 20% test
        split_index = int(0.8 * len(skel_files))
        train_files = skel_files[:split_index]
        test_files = skel_files[split_index:int(len(skel_files))]
        
         # Process and save files for train set
        for file_index, file in tqdm(enumerate(train_files), total=len(train_files), desc=f"Processing training set for {subdir}"):
            file_path = os.path.join(subdir_path, file)
            mat = loadmat(file_path)
            output_file = os.path.join(train_subdir, f'{file_index}.npy') 
            mixed = mat['Data']
            config = load_config('config.json')['signal_processing']
            radar_params = load_config('config.json')['radar_parameters']
            adc_sample = radar_params['numADC']
            chirp_num = int(len(mixed) / adc_sample)
            sx2 = generate_spectrogram(mixed, config, adc_sample , None, chirp_num)
            sweep_time = 3.125e-4
            nfft = config['nfft']
            c = radar_params['c']
            BW = radar_params['BW']
            start_freq = radar_params['start_freq']
            end_freq = start_freq + BW
            fc = (start_freq + end_freq) / 2
            lambda_ = c / fc
            NPpF = radar_params['NPpF']
            frameDuration = radar_params['frameDuration']
            T = frameDuration / NPpF
            Vmax = lambda_ / (T * 4) 
            timeAxis = sweep_time /256 * np.linspace(0, len(mixed)-1,  len(mixed))
            freqAxis = np.linspace(-Vmax / 2, Vmax / 2, nfft)
            result= {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
            np.save(output_file, result)


        # Process and save files for test set
        for file_index, file in tqdm(enumerate(test_files), total=len(test_files), desc=f"Processing test set for {subdir}"):
            file_path = os.path.join(subdir_path, file)
            mat = loadmat(file_path)
            output_file = os.path.join(test_subdir, f'{file_index}.npy')
            mixed = mat['Data']
            config = load_config('config.json')['signal_processing']
            radar_params = load_config('config.json')['radar_parameters']
            adc_sample = radar_params['numADC']
            chirp_num =  int(len(mixed) / adc_sample)
            sx2 = generate_spectrogram(mixed, config, adc_sample , None, chirp_num)
            sweep_time = 3.125e-4
            nfft = config['nfft']
            c = radar_params['c']
            BW = radar_params['BW']
            start_freq = radar_params['start_freq']
            end_freq = start_freq + BW
            fc = (start_freq + end_freq) / 2
            lambda_ = c / fc
            NPpF = radar_params['NPpF']
            frameDuration = radar_params['frameDuration']
            T = frameDuration / NPpF
            Vmax = lambda_ / (T * 4) 
            timeAxis = sweep_time /256 * np.linspace(0, len(mixed)-1,  len(mixed))
            freqAxis = np.linspace(-Vmax / 2, Vmax / 2, nfft)
            result= {'spec': sx2, 'time_axis': timeAxis, 'freq_axis': freqAxis}
            np.save(output_file, result)
