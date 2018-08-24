import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("imagesetfile", help="Input image set file (hdf5 format)")
parser.add_argument("splitindex", type=int, help="0-based index of last image to put in first output file")
parser.add_argument("firstfile", help="First output image set file (hdf5 format)")
parser.add_argument("secondfile", help="Second output image set file (hdf5 format)")

args = parser.parse_args()

infile = h5py.File(args.imagesetfile, 'r')
firstfile = h5py.File(args.firstfile, 'w')
secondfile = h5py.File(args.secondfile, 'w')

in_image_num = 0

while True:
    image = infile.get(str(in_image_num))
    if image is None:
        break
    print("Writing image "+str(in_image_num)+"...")
    if in_image_num <= args.splitindex:
        n = in_image_num
        firstfile.create_dataset(str(n), data=np.array(image, dtype=np.float_))
    else:
        n = in_image_num - (args.splitindex+1)
        secondfile.create_dataset(str(n), data=np.array(image, dtype=np.float_))
    in_image_num += 1
