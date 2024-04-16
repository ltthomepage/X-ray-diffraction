import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Define path for XRD profile
file_path = '1.dat'

# Reading XRD profile
xrd_profile = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=['2Theta', 'Intensity'])
x_mea = xrd_profile['2Theta']
y_mea = xrd_profile['Intensity']

def smooth_xrd_profile(x, y, window_length=50, polyorder=5):
    smoothed_intensity = signal.savgol_filter(y, window_length=window_length, polyorder=polyorder)  
    return pd.DataFrame({'2Theta': x, 'Intensity': smoothed_intensity})

smoothed_xrd_profile = smooth_xrd_profile(x_mea, y_mea)
x_smo = smoothed_xrd_profile['2Theta']
y_smo = smoothed_xrd_profile['Intensity']

def peak_finding(data, threshold_intensity, distance_min, prominence_threshold, width_range):
    x = data['2Theta']
    y = data['Intensity']
    
    peaks, _ = signal.find_peaks(y, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)

    return peaks

peaks = peak_finding(smoothed_xrd_profile, 2000, 500, 10.0, (30, 100))

# Plot original and smoothed XRD profile
plt.figure()
plt.plot(xrd_profile['2Theta'], xrd_profile['Intensity'], label='Original')
plt.plot(smoothed_xrd_profile['2Theta'], smoothed_xrd_profile['Intensity'], label='Smoothed')
plt.plot(smoothed_xrd_profile['2Theta'].iloc[peaks], smoothed_xrd_profile['Intensity'].iloc[peaks], 'ro', label='Peaks')

plt.legend()
plt.show()