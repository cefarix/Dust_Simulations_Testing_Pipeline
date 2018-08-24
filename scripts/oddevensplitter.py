import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("imagesetfile", help="Input image set file (hdf5 format)")
parser.add_argument("evenfile", help="Output image set file of even numbered images (hdf5 format)")
parser.add_argument("oddfile", help="Output image set file of odd numbered (hdf5 format)")

args = parser.parse_args()

infile = h5py.File(args.imagesetfile, 'r')
evenfile = h5py.File(args.evenfile, 'w')
oddfile = h5py.File(args.oddfile, 'w')

in_image_num = 0
even_image_num = 0
odd_image_num = 0

while True:
    image = infile.get(str(in_image_num))
    if image is None:
        break
    print("Writing image "+str(in_image_num)+"...")
    if (in_image_num%2) == 0:
        evenfile.create_dataset(str(even_image_num), data=np.array(image, dtype=np.float_))
        even_image_num += 1
    else:
        oddfile.create_dataset(str(odd_image_num), data=np.array(image, dtype=np.float_))
        odd_image_num += 1
    in_image_num += 1
