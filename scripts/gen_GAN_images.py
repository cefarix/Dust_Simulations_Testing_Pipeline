import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import h5py
from kgan import KGAN

parser = argparse.ArgumentParser()
parser.add_argument("dustcutsfile", help="Dust cuts file to use for denormalization (hdf5 format)")
parser.add_argument("numimages", type=int, help="Number of images to generate.")
parser.add_argument("width", type=int, help="Image width.")
parser.add_argument("height", type=int, help="Image height.")
parser.add_argument("outfile", help="Output filename (hdf5 format)")

args = parser.parse_args()

print("Reading denormalization constants from dust cuts file...")
dustcutsf = h5py.File(args.dustcutsfile, "r")
lognorminfo = dustcutsf.get("lognorminfo")
musigmainfo = dustcutsf.get("musigmainfo")
if (lognorminfo is None) or (musigmainfo is None):
    print("Denormalization constants not found in dust cuts file.")
    print("The denormalization constants will be calculated now and stored in the dust cuts file for future use.")
    print("This only needs to be done once per dust cuts file.")
    dustcutsf.close()
    dustcutsf = h5py.File(args.dustcutsfile, "r+")
    n = 0
    lognorm_min = 1e100
    lognorm_max = 1e-100
    means = []
    squared_means = []
    while True:
        image = dustcutsf.get(str(n))
        if image is None:
            break
        print("Processing image", n+1, "in dust cuts file...")
        image = np.array(image)
        mean = np.mean(image)
        squared_mean = np.mean(image**2.0)
        means.append(mean)
        squared_means.append(squared_mean)
        logimage_min = np.log(np.amin(image))
        logimage_max = np.log(np.amax(image))
        lognorm_min = min(lognorm_min, logimage_min)
        lognorm_max = max(lognorm_max, logimage_max)
        n += 1
    mu = np.mean(means)
    sigma = np.sqrt(np.mean(squared_means) - mu)
    print("Storing denormalization constants in dust cuts file...")
    dustcutsf.create_dataset("lognorminfo", data=np.array([lognorm_min, lognorm_max]))
    dustcutsf.create_dataset("musigmainfo", data=np.array([mu, sigma]))
else:
    lognorm_min = lognorminfo[0]
    lognorm_max = lognorminfo[1]
dustcutsf.close()

print("Creating output file...")
outf = h5py.File(args.outfile, "w")
outf.create_dataset("lognorminfo", data=np.array([lognorm_min, lognorm_max]))

print("Loading GAN...")
gan = KGAN(args.width, args.height, load_dir='./')
noise = np.random.normal(loc=0., scale=1., size=[args.numimages, 64])
batch_size = 16
imagenum = 0
means = []
squared_means = []
while imagenum < args.numimages:
    lower_bound = imagenum
    upper_bound = imagenum + batch_size
    if upper_bound > args.numimages:
        upper_bound = args.numimages
    print("Generating images", lower_bound+1, "through", upper_bound, "...")
    images = gan.G.predict(noise[lower_bound:upper_bound,:])
    for n in np.arange(lower_bound, upper_bound):
        print("Writing image", n+1, "of", args.numimages,"...")
        image = np.array(images[n-lower_bound,:,:,0])
        denorm_image = np.exp((image+1.0)*(lognorm_max-lognorm_min)*0.5+lognorm_min)
        mean = np.mean(denorm_image)
        squared_mean = np.mean(denorm_image**2.0)
        means.append(mean)
        squared_means.append(squared_mean)
        outf.create_dataset(str(n), data=denorm_image)
    outf.flush()
    imagenum = upper_bound
mu = np.mean(means)
sigma = np.sqrt(np.mean(squared_means) - mu)
outf.create_dataset("musigmainfo", data=np.array([mu, sigma]))

