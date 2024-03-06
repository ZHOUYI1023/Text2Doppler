import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
from scipy.signal import butter, lfilter
from numba import jit


def load_config(filename):
    """
    Loads configuration from a JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)
        
        
class RadarRCSProcessor:
    def __init__(self, tar_loc, tx_loc, rx_loc, powerTx, numChirps):
        self.tar_loc = tar_loc
        self.tx_loc = tx_loc
        self.rx_loc = rx_loc
        self.powerTx = powerTx
        self.numChirps = numChirps
        self.amp_all = np.zeros([tar_loc.shape[0],numChirps])
	
    @staticmethod
    @jit(nopython=True)
    def rcs_ellipsoid(a, b, c, phi, theta):
        """
        Calculate the radar cross section (RCS) of an ellipsoid.
        """
        nomi = np.pi * a**2 * b**2 * c**2
        denomi = ((a**2) * (np.sin(theta)**2) * (np.cos(phi)**2) +
                  (b**2) * (np.sin(theta)**2) * (np.sin(phi)**2) +
                  (c**2) * (np.cos(theta)**2))**2
        return nomi / denomi
        
    @staticmethod
    @jit(nopython=True)
    def calculate_angles(tx_loc, body_part, aspect_vector, r_dist):
        """
        Calculate theta and phi angles for radar processing.
        """
        A = tx_loc - body_part
        A_dot_aspect_vector = np.dot(A, aspect_vector)
        norm_A = np.sqrt(np.sum(A**2))
        norm_aspect_vector = np.sqrt(np.sum(aspect_vector**2))
        theta_angle = np.arccos(A_dot_aspect_vector / (norm_A * norm_aspect_vector))
        phi_angle = np.arcsin((tx_loc[1] - body_part[1]) / np.sqrt(r_dist[0]**2 + r_dist[1]**2))
        return theta_angle, phi_angle

    def process_body_part(self, target_id, end_part_index, ellipsoid_params):
        """
        Process radar returns for a specific body part.
        """
        body_part = np.zeros([3, self.numChirps])
        ref_point = np.zeros([3, self.numChirps])
        #amp = np.zeros([len(target_id), self.numChirps])

        for k in range(self.numChirps):
            body_part[:, k] = self.tar_loc[end_part_index[0], :, k]
            ref_point[:, k] = self.tar_loc[end_part_index[1], :, k]
            body_part_length = np.sqrt(np.sum((body_part[:, k] - ref_point[:, k]) ** 2))

            r_dist = np.abs(body_part[:, k] - self.tx_loc[0].T)
            dist_tx = np.sqrt(np.sum(r_dist ** 2, axis=0))

            aspect_vector = body_part[:, k] - ref_point[:, k]
            theta_angle, phi_angle = self.calculate_angles(self.tx_loc[0], body_part[:, k], aspect_vector, r_dist)
            a, b = ellipsoid_params
            c = body_part_length / 2 # Update c based on the body part length
            rcs = self.rcs_ellipsoid(a, b, c, phi_angle, theta_angle)
            dist_rx = np.sqrt(np.sum((body_part[:, k] - self.rx_loc[0].T) ** 2))
            self.amp_all[target_id, k] = np.sqrt(rcs * self.powerTx) / (dist_tx * dist_rx)


def generate_spectrogram(mixed, config, numADC, NPpF, numChirps):
    # Extract parameters from config
    numTX = config['numTX']
    numRX = config['numRX']
    nfft = config['nfft']
    window = config['window']
    noverlap = config['noverlap']
    order = config['butterworth_order']
    cutoff = config['butterworth_cutoff']

    # Reshape and permute RDC
    RDC = mixed.reshape(numTX * numRX, numADC, numChirps, order="F")
    RDC = np.transpose(RDC, (1, 2, 0))

    rBin = np.arange(numADC)
    shift = window - noverlap

    # Summing along the antenna dimension and applying the FFT
    range_profile = np.fft.fft(RDC[rBin, :, 0], axis=0)

    # Butterworth filter parameters
    b, a = butter(order, cutoff, 'high')
    ns = range_profile.shape[1]
    mti_filtered = np.zeros_like(range_profile)
    for k in range(range_profile.shape[0]):
        mti_filtered[k, :ns] = lfilter(b, a, range_profile[k, :ns])

    # Remove the first row from the filtered data
    mti_filtered = mti_filtered[1:, :]
    RDC_summed = np.sum(mti_filtered, axis=0)

    N = int((len(RDC_summed) - window - 1) / shift)
    # Initialize the output array for spectrogram
    sx = np.zeros((nfft, N), dtype=complex)

    for i in range(N):
        start_index = i * shift
        end_index = start_index + window
        segment = RDC_summed[start_index:end_index]
        windowed_segment = segment * np.hanning(window)  # Apply Hann window
        fft_result = np.fft.fft(windowed_segment, nfft)  # FFT
        sx[:, i] = fft_result

    # Post-processing for visualization
    sx2 = np.abs(np.fft.fftshift(np.flipud(sx), axes=0))

    return sx2

   
def process_spectrogram(skel_hist, new_origin, rcs_flag = True):
    """
    Processes the skeleton history to generate a radar spectrogram.
    
    Parameters:
    - skel_hist: numpy array containing the skeleton history.
    
    Returns:
    - sx2: The generated spectrogram.
    """
    # Load radar configuration
    config = load_config('config.json')
    radar_params = config['radar_parameters']

    # Extract radar parameters from configuration
    c = radar_params['c']
    BW = radar_params['BW']
    start_freq = radar_params['start_freq']
    end_freq = start_freq + BW
    fc = (start_freq + end_freq) / 2
    numADC = radar_params['numADC']
    NPpF = radar_params['NPpF']
    frameDuration = radar_params['frameDuration']
    T = frameDuration / NPpF
    PRF = 1 / T
    F = numADC / T
    dt = 1 / F
    slope = BW / T
    lambda_ = c / fc

    # Define radar constants and arrays
    t_onePulse = np.arange(0, dt * numADC, dt)
    numTX = 1
    numRX = 1
    Vmax = lambda_ / (T * 4)  # Max Unambiguous velocity m/s
    DFmax = 1 / 2 * PRF  # Max Unambiguous Doppler Frequency
    dR = c / (2 * BW)  # range resolution
    Rmax = F * c / (2 * slope)  # Max range

    d_rx = lambda_ / 2  # distance between RXs
    d_tx = 4 * d_rx  # distance between TXs

    # Calculate antenna locations based on the new_origin and distances between antennas
    radar_loc_bias = np.array([new_origin[0], new_origin[2], new_origin[1]])
    tx_loc = [np.array([(i) * d_tx, 0, 0]) + radar_loc_bias for i in range(numTX)]
    rx_loc = [np.array([tx_loc[-1][0] + d_tx + (i) * d_rx, 0, 0]) + radar_loc_bias for i in range(numRX)]

    # Process skeleton history to calculate target locations and velocities
    fps_skel = 20
    num_tar = skel_hist.shape[0]  # Number of targets
    durationx = skel_hist.shape[2] / fps_skel
    numChirps = int(durationx * NPpF * (1 / frameDuration))
    numCPI = numChirps // NPpF

    tar_loc = np.zeros((num_tar, skel_hist.shape[1], numChirps))
    vel_hist = np.zeros((num_tar, skel_hist.shape[1]))

    for t in range(num_tar):
        for i in range(skel_hist.shape[1]):
            x = np.linspace(1, skel_hist.shape[2], numChirps)
            cs = CubicSpline(np.arange(1, skel_hist.shape[2] + 1), skel_hist[t, i, :])
            tar_loc[t, i, :] = cs(x)
            vel_hist[t, i] = (np.max(tar_loc[t, i, :]) - np.min(tar_loc[t, i, :])) * np.sqrt(3) / durationx
            
    # TX
    delays_targets = np.empty((numTX, numRX, num_tar), dtype=object)

    for t in range(num_tar):
        for i in range(numTX):
            for j in range(numRX):
                # Reshape tar_loc to 2D array: (numChirps, 3)
                tar_loc_rep = tar_loc[t, :, :].T
                # Calculate distances
                dist_rx = cdist(tar_loc_rep, rx_loc[j][np.newaxis, :])
                dist_tx = cdist(tar_loc_rep, tx_loc[i][np.newaxis, :])
                delays_targets[i, j, t] = (dist_rx + dist_tx) / c
                
    if rcs_flag:            
        # Initialize the RadarRCSProcessor with target locations and antenna configurations
        rcs_processor = RadarRCSProcessor(tar_loc, tx_loc, rx_loc, radar_params['powerTx'], numChirps)
        # Load body part configuration and process RCS for each body part
        body_part_config = load_config('body_part.json')
        for part in body_part_config['body_parts']:
            rcs_processor.process_body_part(
                target_id=part["target_id"],
                end_part_index=part["end_part_index"],
                ellipsoid_params=part["ellipsoid_params"]
            )

        amp = rcs_processor.amp_all  # Amplitude of the RCS for all targets
        # print(amp[:,0])

    # Define phase functions
    phase = lambda tx, fx: 2 * np.pi * (fx * tx + slope / 2 * tx**2)
    phase2 = lambda tx, fx, r, v: 2 * np.pi * (2 * fx * r / c + tx * (2 * fx * v / c + 2 * slope * r / c))

    phase_t = phase(t_onePulse, fc)

    # Initialize the mixed signal array
    mixed = np.zeros((numTX, numRX, numChirps * numADC), dtype=complex)
    excluded_ind = [5,10,16,21,26]

    for i in range(numTX):
        for j in range(numRX):
            #print(f'Processing Channel: {j+1}/{numRX}')
            for t in range(num_tar):
                if t not in excluded_ind:
                    #print(f'{t+1}/{num_tar}')
                    signal_tar = np.zeros(numChirps * numADC, dtype=complex)
                    for k in range(numChirps):
                        phase_tar = phase(t_onePulse - delays_targets[i, j, t][k], fc)  # received
                        signal_tar[k*numADC:(k+1)*numADC] = np.exp(1j * (phase_t - phase_tar))
                if rcs_flag:
                    mixed[i, j, :] += np.repeat(amp[t, :], numADC) * signal_tar.conj()
                else: 
                    mixed[i, j, :] += signal_tar
    # Generate the spectrogram using the processed radar data and signal processing configuration
    signal_processing_config = load_config('config.json')['signal_processing']
    sx2 = generate_spectrogram(mixed, signal_processing_config, numADC, NPpF, numChirps)
    timeAxis = np.linspace(0, numCPI * frameDuration, numCPI)
    freqAxis = np.linspace(-Vmax / 2, Vmax / 2, signal_processing_config['nfft'])
    return sx2, timeAxis, freqAxis
