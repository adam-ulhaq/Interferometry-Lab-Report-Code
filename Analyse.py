# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:13:16 2023
This version developed by Laura Hollister
Edited and enhanced by Elias Fink

Correct mercury lamp spectrum using green line with known wavelength.
Change parameters throughout script to fit parameters of experimental setup.
"""
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.interpolate as spi
from scipy.optimize import curve_fit
import sys
#############################################################################
file = "Interferometry/Data/Hg Lamp Full Spectrum Data.txt" #Choose file here

def main():
    # -- Step 1: Import data (y_data: yellow doublet / full spectrum, y_ref: green line) --
    y_data, y_ref = np.loadtxt(file, unpack=True, usecols=(0, 1)) # change order of args to apply to detector order

    # -- Step 1.1: Set reference wavelength (green line) --
    ref_wl = 546/2  # factor 2 as each crossing = half lambda (546 for green line, 633 for HeNe laser)
    sampling_freq = 50
    x = 0.7 * np.arange(0, len(y_ref), 1)/sampling_freq # distance travelled by motor
    
    a=2
    x=x[a:len(x)-a]# deletes the first motion of motor as it tuen=rs on
    y_data=y_data[a:len(y_data)-a]
    y_ref=y_ref[a:len(y_ref)-a]
    
    # -- Step 1.2: Remove offset and Butterworth filter --
    y_ref -= y_ref.mean()
    y_data -= y_data.mean()
    
    freq = 1  # cutoff frequency
    sos = signal.butter(2, freq, 'hp', fs=sampling_freq, output='sos')
    y_ref = signal.sosfilt(sos, y_ref)  # filter the y values
    y_data = signal.sosfilt(sos, y_data)

    # -- Step 2: Find the crossing points --
    crossing_points = np.array([])
    for i in range(len(y_ref)-1):
        if (y_ref[i] <= 0 and y_ref[i+1] >= 0) or (y_ref[i] >= 0 and y_ref[i+1] <= 0): # check for sign change in y
            b = (x[i] * y_ref[i+1] - x[i+1] * y_ref[i]) / (x[i] - x[i+1])
            a = (y_ref[i+1] - b)/x[i+1]
            extra = -b/a - x[i]
            crossing_points = np.append(crossing_points, (x[i]+extra)) # interpolate crossing point

    # -- Step 2.1: Plot crossing points --
    if True: # set to false if you do not want to see this
        plt.figure("Find the crossing points")
        plt.title("Find the crossing points")
        plt.plot(x, y_ref, 'x-', label='Data')
        plt.plot(crossing_points, 0*np.array(crossing_points),'go', label='Crossing Points')
        plt.xlabel('Distance')
        plt.ylabel('Intensity, a.u.')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # -- Step 3: Shift the points so there is equal spacing --
    index = 0
    x_correct_array = []
    last_pt = 0
    last_pt_correct = 0

    for period in range(len(crossing_points)//2-1):
        measured_lam = crossing_points[2*period+2] - crossing_points[2*period] # distance between crossing points x 2 -> corresponding to lambda/2
        shifting_ratio = ref_wl / measured_lam # ratio between lambda/2 and distance on interferogram corresponding to lambda/2
        while x[index] < crossing_points[2*period+2]:
            x_correct = shifting_ratio * (x[index] - last_pt) + last_pt_correct # stretching whole interferogram by shifting_ratio in x-direction
            x_correct_array.append(x_correct)
            index += 1
        last_pt = x[index-1]
        last_pt_correct = x_correct_array[-1]

    # -- Step 4: Cubic spline interpolation --
    N = int(1e6)  # number of data points resampled
    x_spline = np.linspace(0, x_correct_array[-1], N)  # x vals of spline points
    y = y_data[:len(x_correct_array)]
    cs = spi.CubicSpline(x_correct_array, y)  # cubic spline y vals

    # -- Step 4.1: Plot initial and interpolated points --
    if True: # set to false if you do not want to see this
        plt.figure("Correct the points and resample the points", figsize=(8, 6))
        plt.title('Corrected and Resampled Interferogram')
        plt.plot(x_correct_array, y, 'go', label='Inital points')
        plt.plot(x_spline, cs(x_spline), label="Cubic Spline")
        plt.ylabel('Intensity, a.u.')
        plt.xlabel('Distance travelled, µsteps')
        plt.legend()
        plt.grid()
        plt.show()

    # -- Step 5: FFT to extract spectra --
    distance = x_spline[1:]-x_spline[:-1]  # distance travelled
    xf1 = spf.fftshift(spf.fftfreq(len(x_spline))) # fourier transform and shift
    xvals = abs(2*distance.mean()/xf1[int(len(xf1)/2+1):len(xf1)]) # go from k -> lambda and scale x to nm

    yf1 = spf.fftshift(spf.fft(cs(x_spline))) # fourier transform and shift
    yvals = abs(yf1[int(len(xf1)/2+1):len(xf1)]) # only taking values for +ve x
    yvals = yvals/max(yvals) # normalise intensity

    # -- Step 5.1: Find peaks in spectrum and plotfit gaussian of peak --
    peaks = []
    sigmas = []

    #Extra Addition by A.M.U., Fit if peak is further than 5 stds from mean
    mean_intensity = np.mean(yvals)
    std_intensity = np.std(yvals)
    y_lower = mean_intensity + 5*std_intensity 
    y_lower = 0.1

    for i in range(len(yvals)):
        if yvals[i] > y_lower and yvals[i] > yvals[i-1] and yvals[i] > yvals[i+1]: # peak defined as any local maximum at intensity > 0.1
            peaks.append(xvals[i])
            A = yvals[i] # amplitude is peak value
            mu = xvals[i] # centre is peak position
            sigma, _ = curve_fit(lambda x, sigma: A*np.exp(-(x-mu)**2/(2*sigma**2)), xvals[i-5:i+5], yvals[i-5:i+5], p0 = [0.1]) # curve fitting Gaussian
            sigmas.append([A, mu, sigma[0]])
    #print(sigmas) # use this print if you want to use the values for data analysis

    # -- Step 5.2: Plot corrected spectra --
    plt.figure("Fully corrected spectrum FFT")
    plt.title('Mercury Spectrum')
    plt.plot(xvals, yvals, label='Data')
    plt.plot(peaks, np.zeros(len(peaks)), 'o', label='Peak positions')

    # -- Step 5.3: Plot fitted gaussians of peaks --
    xfit = np.array([])
    for peak in peaks: # higher resolution x-inputs around peaks to see Gaussians
        xfit = np.append(xfit, np.linspace(peak - 10, peak + 10, 10000))
    plt.plot(xfit, gauss(xfit, sigmas), '-', label='Fitted Gaussians')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Arbitrary Units)')
    plt.xlim(0, 1200)  # limits to zoom into part of spectrum (400, 600 for full mercury spectrum / 570, 585 for yellow doublet)
    plt.grid()
    plt.legend()
    plt.show()

    print(peaks)
    print(sigmas)
    print(np.mean(yvals))
    print(np.std(yvals))

    # Adam Data Analysis:
    ''' Was broken so retried this a second time later.
    As0 = [0.06, 0.3, 0.06, 0.06]
    mus0 = [435, 545, 577, 579]
    sigs0 = [0.1, 0.1, 0.1, 0.1]
    C0 = np.mean(yvals)

    
    p0 = [As0, mus0, sigs0, C0]

    As, mus, sigs, C = curve_fit(Super_func, xvals, yvals, p0=p0)
    

    plt.figure("Fully corrected spectrum FFT")
    plt.title('Mercury Spectrum')
    plt.plot(xvals, yvals, label='Data')

    for i in range (0, 4):
        A, mu, sig = curve_fit(Lorentzian, xvals, yvals, p0=[As0[i], mus0[i], sigs0[i]])
        plt.plot(xvals, Lorentzian(xvals, A, mu, sig))
        print(i)

    
    #plt.plot(peaks, np.zeros(len(peaks)), 'o', label='Peak positions')
    #plt.plot(Super_func(xvals, As, mus, sigs, C))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim(0, 1000)  # limits to zoom into part of spectrum (400, 600 for full mercury spectrum / 570, 585 for yellow doublet)
    plt.grid()
    plt.legend()
    plt.show()
'''

    #Adam's Data Analysis V2
    peaks = []
    sigs = []

    for i in range(len(yvals)):
        if yvals[i] > y_lower and yvals[i] > yvals[i-1] and yvals[i] > yvals[i+1]: # peak defined as any local maximum at intensity > 0.1
            peaks.append(xvals[i])
            A = yvals[i] # amplitude is peak value
            mu = xvals[i] # centre is peak position
            sig, _ = curve_fit(lambda x, sig: A*(sig/np.pi)*1/((x - mu)**2 + sig**2), xvals, yvals, p0 = [0.1]) # curve fitting Gaussian
            sigs.append([A, mu, sig[0]])

    plt.figure("Fully corrected spectrum FFT")
    plt.title('Mercury Spectrum')
    plt.plot(xvals, yvals, label='Data')
    plt.plot(peaks, np.zeros(len(peaks)), 'o', label='Peak positions')

    xfit = np.array([])
    for peak in peaks: # higher resolution x-inputs around peaks to see Gaussians
        xfit = np.append(xfit, np.linspace(peak - 10, peak + 10, 10000))
    
    for i in range(0, len(sigs)):
        plt.plot(xfit, Lorentzian(xfit, *sigs[i]), '-', label='Fitted Lorentzian', color='green')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Arbitrary Units)')
    plt.xlim(0, 1000)  # limits to zoom into part of spectrum (400, 600 for full mercury spectrum / 570, 585 for yellow doublet)
    plt.grid()
    plt.legend()
    plt.show()

    print(sigs)


def gauss(x, arr):
    # returns sum of all fitted Gaussian peaks
    val = 0
    A = [arr[i][0] for i in range(len(arr))]
    mu = [arr[i][1] for i in range(len(arr))]
    sigma = [arr[i][2] for i in range(len(arr))]
    for i in range(len(arr)):
        val += A[i]*np.exp(-(x-mu[i])**2/(2*sigma[i]**2))
    return val

def Lorentzian(x, A, mu, sig):
    func = A * (sig/np.pi) * (1/((x - mu)**2 + sig**2))
    return func

def Super_func(x, A_arr, mu_arr, sig_arr, const):
    func1 = const
    func2 = Lorentzian(x, A_arr[0], mu_arr[0], sig_arr[0])
    func3 = Lorentzian(x, A_arr[1], mu_arr[1], sig_arr[1])
    func4 = Lorentzian(x, A_arr[2], mu_arr[2], sig_arr[2])
    func5 = Lorentzian(x, A_arr[3], mu_arr[3], sig_arr[3])
    func = func1 + func2 + func3 + func4
    return func

if __name__ == "__main__":
    main()