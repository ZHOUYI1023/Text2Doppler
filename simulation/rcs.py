import numpy as np

class RadarRCSProcessor:
    def __init__(self, tar_loc, tx_loc, rx_loc, powerTx, numChirps):
        self.tar_loc = tar_loc
        self.tx_loc = tx_loc
        self.rx_loc = rx_loc
        self.powerTx = powerTx
        self.numChirps = numChirps
        self.amp_all = np.zeros([tar_loc.shape[0],numChirps])

    def rcs_ellipsoid(self, a, b, c, phi, theta):
        """
        Calculate the radar cross section (RCS) of an ellipsoid.
        """
        nomi = np.pi * a**2 * b**2 * c**2
        denomi = ((a**2) * (np.sin(theta)**2) * (np.cos(phi)**2) +
                  (b**2) * (np.sin(theta)**2) * (np.sin(phi)**2) +
                  (c**2) * (np.cos(theta)**2))**2
        return nomi / denomi

    def calculate_angles(self, tx_loc, body_part, aspect_vector, r_dist):
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
