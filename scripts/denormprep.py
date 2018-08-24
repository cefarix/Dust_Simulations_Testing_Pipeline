import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser(description='Calculates denormalization constants and inserts them into a dust cuts file. Use only on real data.')
parser.add_argument("dustcutsfile", help="Dust cuts file (hdf5 format)")

args = parser.parse_args()

dustcutsf = h5py.File(args.dustcutsfile, 'r+')

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
