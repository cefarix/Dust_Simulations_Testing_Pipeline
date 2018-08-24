import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import h5py
from kgan import KGAN

parser = argparse.ArgumentParser(description="Calculates and saves log-intensity histograms from the GAN model in the current directory. Generates random sets of images from the GAN and calculates the histogram on each set.")
parser.add_argument("dustcutsfile", help="Dust cuts file to use for denormalization.")
parser.add_argument("numsets", type=int, help="Number of image sets to calculate histograms for.")
parser.add_argument("imagesperset", type=int, help="Number of images per set to generate from GAN.")
parser.add_argument("histrangelower", type=float, help="Lower range of histogram bins in log-space.")
parser.add_argument("histrangeupper", type=float, help="Upper range of histogram bins in log-space.")
parser.add_argument("numhistbins", type=int, help="Number of evenly spaced histogram bins in log-space.")
parser.add_argument("width", type=int, help="Image width.")
parser.add_argument("height", type=int, help="Image height.")
parser.add_argument("outfile", help="hdf5 file to which the histograms are saved (see source comments for more format info)")

args = parser.parse_args()

print("Reading denormalization info from dust cuts file...")
dustcutsf = h5py.File(args.dustcutsfile, 'r')
lognorminfo = dustcutsf['lognorminfo']
lognorm_min = lognorminfo[0]
lognorm_max = lognorminfo[1]
dustcutsf.close()

print("Creating output file...")
outf = h5py.File(args.outfile, 'w')

print("Loading GAN...")
gan = KGAN(args.width, args.height, load_dir='./')

"""
A histogram is calculated for each image set. The histogram is saved as a numpy array in outf[str(setnum)].
Histogram bin edges are saved as a numpy array in outf['bins']
"""
batch_size = 16
base_conversion = 1.0/np.log(10.0)
hist_range = (args.histrangelower, args.histrangeupper)
for setnum in range(args.numsets):        
    noise = np.random.normal(loc=0., scale=1., size=[args.imagesperset, 64])
    imagenum = 0
    hist = np.zeros(args.numhistbins)
    while imagenum < args.imagesperset:
        lower_bound = imagenum
        upper_bound = imagenum + batch_size
        if upper_bound > args.imagesperset:
            upper_bound = args.imagesperset
        print("Generating and processing images "+str(lower_bound+1)+" through "+str(upper_bound)+" of set "+str(setnum+1))
        images = gan.G.predict(noise[lower_bound:upper_bound,:])
        # denorm_image = np.exp((image+1.0)*(lognorm_max-lognorm_min)*0.5+lognorm_min)
        # log_image = np.log10(denorm_image)
        log10_denorm_images = ((images[:,:,:,0]+1.0)*(lognorm_max-lognorm_min)*0.5+lognorm_min)*base_conversion
        (imageshist, binedges) = np.histogram(log10_denorm_images, args.numhistbins, hist_range)
        hist += imageshist
        imagenum = upper_bound
    print("Calculated histogram for set "+str(setnum+1)+"/"+str(args.numsets)+".")
    outf.create_dataset(str(setnum), data=hist)
outf.create_dataset("binedges", data=binedges)
outf.create_dataset("imagesize", data=np.array([args.width, args.height]))
outf.create_dataset("datasize", data=np.array([args.numsets, args.imagesperset]))
