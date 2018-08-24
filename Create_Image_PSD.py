from scipy import fftpack
import numpy as np
from ImageTools import getInterpolatedPixelValues

def _azimuthalAverage(image):
    averages = np.array([])
    center = (image.shape - np.array([1.0, 1.0])) / 2.0
    # Take half of the distance from the center to the closest edge
    max_radius = int(min(image.shape) / 2.0 - 1.0)
    # Loop over the radius bins starting from smallest (zero). Each bin is size 1.
    for radius in range(0, max_radius+1):
        # The number of pixels averaged is equal to the circumference+1 of this radius bin
        circumference = 2.0*np.pi*radius
        thetas = np.linspace(0.0, 2*np.pi, circumference+1, False)
        xcoords = radius*np.cos(thetas) + center[1]
        ycoords = radius*np.sin(thetas) + center[0]
        values = getInterpolatedPixelValues(image, xcoords, ycoords)
        avg = np.sum(values) / values.size
        averages = np.append(averages, avg)
    return averages

"""
Calculate the 1d & 2d power spectral density of a square image
1d psd is calculated from 2d psd by starting at the center and averaging radially until the nearest edge
image must be square
"""
def calcSquareImagePSD(image):
    # Get the 2D FFT of the image
    # Then shift it so that the DC component is at the center
    fft2d = fftpack.fftshift( fftpack.fft2(image) )
    
    # Square the magnitude to get the 2D PSD
    psd2d = np.abs(fft2d)**2
    
    # Average azimuthally with ever-increasing radii starting at the center and towards the corners
    # This averages in a circle at the center of the image
    psd1d = _azimuthalAverage(psd2d)
    
    return (psd1d, psd2d)
