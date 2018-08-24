import argparse
import numpy as np
from Generate_Stats import StatMaker

parser = argparse.ArgumentParser(description="Updates a StatMaker cache file. This can be used to calculate or re-calculate missing or selected statistics on all image batches.")
parser.add_argument("imagesetfile", help="Image set file (read-only). (hdf5 format)")
parser.add_argument("smcachefile", help="Cached statistics file (read-write, created if it doesn't exist). (hdf5 format)")
parser.add_argument("batchsize", type=int, help="Number of images per batch.")
parser.add_argument("statistics", help="Comma separated list of statistics to calculate ('psd', 'intensityhistogram', and 'mf2d') or 'missing'. When 'missing' is specified only missing statistics are calculated and existing statistics are not recalculated. See StatMaker's recalculateStatisticsOnBatches method for more info.")

args = parser.parse_args()

sm = StatMaker(args.smcachefile, '', args.imagesetfile, args.batchsize)
if args.statistics.strip() == 'missing':
    sm.calculateStatistics()
else:
    sm.recalculateStatisticsOnBatches(None, [s.strip() for s in args.statistics.split(",") if s.strip()!=''])
