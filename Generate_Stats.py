import numpy as np
import h5py
from Create_Image_PSD import calcSquareImagePSD
from minkfncts2d import MF2D

class _ImageBatch:
    def __init__(self):
        self.psds = None
        self.logpsds = None
        self.logintensityhistogram = None
        self.mf2d_F = None
        self.mf2d_U = None
        self.mf2d_Chi = None

class StatMaker:
    def __init__(self, smcache_h5filename, imagesetname, imageset_h5filename, images_per_batch):
        self.imagesetname = imagesetname
        
        self.imageset_file = h5py.File(imageset_h5filename, 'r')
        lognorminfo = self.imageset_file.get('lognorminfo')
        if lognorminfo is None:
            raise KeyError('lognorminfo not found in image set file!')
        self.lognorm_min = lognorminfo[0]
        self.lognorm_max = lognorminfo[1]
        musigmainfo = self.imageset_file.get('musigmainfo')
        if musigmainfo is None:
            raise KeyError('musigmainfo not found in image set file!')
        self.image_set_mu = musigmainfo[0]
        self.image_set_sigma = musigmainfo[1]
        
        self.images_per_batch = images_per_batch
        self.image_batches = {}

        self.num_logintensityhistogram_bins = 100
        self.logintensityhistogram_range = [-5.0, 10.0]
        self.logintensityhistogram_binedges = np.linspace(self.logintensityhistogram_range[0], self.logintensityhistogram_range[1],self.num_logintensityhistogram_bins+1)

        self.num_mf2d_thresholds = 100
        self.mf2d_thresholds = np.linspace(-1.0, 1.0, self.num_mf2d_thresholds)
        
        self.currentBatchNumber = -1
        self.num_loaded_images = 0
        self._loadCached(smcache_h5filename)
        
    def _loadCached(self, smcache_h5filename):
        self.smcache = h5py.File(smcache_h5filename, 'a')
        self.smcache_spectra = self.smcache.require_group("spectra")
        self.smcache_logspectra = self.smcache.require_group("logspectra")
        self.smcache_logintensityhistograms = self.smcache.require_group("logintensityhistograms")
        self.smcache_mf2d = self.smcache.require_group("mf2d")
            
    def _loadAndCalcImageBatchCachedInfo(self, batchNumber, use_only_cached=False):
        if batchNumber in self.image_batches:
            return self.image_batches[batchNumber]
        
        psds = self.smcache_spectra.get(str(batchNumber))
        logpsds = self.smcache_logspectra.get(str(batchNumber))
        hist = self.smcache_logintensityhistograms.get(str(batchNumber))
        mf2d_data = self.smcache_mf2d.get(str(batchNumber))
        images = []

        load_images = False
        if (psds is None) or (logpsds is None) or (hist is None) or (mf2d_data is None):
            load_images = True
        if load_images and use_only_cached:
            return None
        if load_images:
            print("Batch", batchNumber, ": Loading images.")
            imageNumber = batchNumber * self.images_per_batch
            for i in np.arange(self.images_per_batch):
                image = self.imageset_file.get(str(imageNumber))
                if image is None:
                    break
                images.append(image)
                imageNumber += 1
            if len(images) == 0:
                return None
            images = np.array(images)

        if psds is None:
            # Assume all images are square and same dimensions
            print("Batch", batchNumber, ": Calculating PSDs.")
            psds = []
            logpsds = []
            for image in images:
                (psd, _) = calcSquareImagePSD(image)
                logpsd = np.log10(np.clip(psd, 1.0, None))
                psds.append(psd)
                logpsds.append(logpsd)
            psds = np.array(psds, dtype=np.float64)
            logpsds = np.array(logpsds, dtype=np.float64)
            self.smcache_spectra.create_dataset(str(batchNumber), data=psds)
            self.smcache_logspectra.create_dataset(str(batchNumber), data=logpsds)
        else:
            print("Batch", batchNumber, ": PSDs are cached.")
        
        if hist is None:
            print("Batch", batchNumber, ": Calculating pixel intensity histogram.")
            (hist, _) = np.histogram(np.log10(images), self.num_logintensityhistogram_bins, range=self.logintensityhistogram_range)
            self.smcache_logintensityhistograms.create_dataset(str(batchNumber), data=hist)
        else:
            print("Batch", batchNumber, ": Pixel intensity histogram is cached.")
        
        if mf2d_data is None:
            print("Batch", batchNumber, ": Calculating 2D Minkowski functionals.")
            mf2d_F = []
            mf2d_U = []
            mf2d_Chi = []
            for image in images:
                #standardized_image = (np.array(image, dtype=np.float_) - self.image_set_mu) / self.image_set_sigma
                log_norm_image = np.array(2.0*(np.log(image) - self.lognorm_min)/(self.lognorm_max - self.lognorm_min) - 1.0, dtype=np.float_)
                mf2d_f = []
                mf2d_u = []
                mf2d_chi = []
                for t in self.mf2d_thresholds:
                    (f, u, chi) = MF2D(log_norm_image, t)
                    mf2d_f.append(f)
                    mf2d_u.append(u)
                    mf2d_chi.append(chi)
                mf2d_F.append(mf2d_f)
                mf2d_U.append(mf2d_u)
                mf2d_Chi.append(mf2d_chi)
            mf2d_F = np.array(mf2d_F)
            mf2d_U = np.array(mf2d_U)
            mf2d_Chi = np.array(mf2d_Chi)
            mf2d_data = np.array([mf2d_F, mf2d_U, mf2d_Chi])
            self.smcache_mf2d.create_dataset(str(batchNumber), data=mf2d_data)
        else:
            print("Batch", batchNumber, ": 2D Minkowski functionals are cached.")
        
        if load_images:
            self.smcache.flush()
        
        batch = _ImageBatch()
        batch.psds = np.array(psds)
        batch.logpsds = np.array(logpsds)
        batch.logintensityhistogram = np.array(hist)
        batch.mf2d_F = np.array(mf2d_data[0])
        batch.mf2d_U = np.array(mf2d_data[1])
        batch.mf2d_Chi = np.array(mf2d_data[2])

        self.image_batches[batchNumber] = batch
        return batch
    
    def _calcLogSpectraMeanAndStdDev(self):
        self.logspectra = []
        for _, b in self.image_batches.items():
            self.logspectra.extend(b.logpsds)
        self.logspectra_mean = np.mean(self.logspectra, axis=0)
        self.logspectra_stddev = np.std(self.logspectra, axis=0)
    
    def _calcMF2DMeanAndStdDev(self):
        self.mf2d_F = []
        self.mf2d_U = []
        self.mf2d_Chi = []
        for _, b in self.image_batches.items():
            self.mf2d_F.extend(b.mf2d_F)
            self.mf2d_U.extend(b.mf2d_U)
            self.mf2d_Chi.extend(b.mf2d_Chi)
        self.mf2d_F = np.array(self.mf2d_F)
        self.mf2d_U = np.array(self.mf2d_U)
        self.mf2d_Chi = np.array(self.mf2d_Chi)
        self.mf2d_F_mean = np.mean(self.mf2d_F, axis=0)
        self.mf2d_U_mean = np.mean(self.mf2d_U, axis=0)
        self.mf2d_Chi_mean = np.mean(self.mf2d_Chi, axis=0)
        self.mf2d_F_stddev = np.std(self.mf2d_F, axis=0)
        self.mf2d_U_stddev = np.std(self.mf2d_U, axis=0)
        self.mf2d_Chi_stddev = np.std(self.mf2d_Chi, axis=0)
    
    def calculateCumulativeStatisticsForNextBatch(self, use_only_cached=False):
        batch = self._loadAndCalcImageBatchCachedInfo(self.currentBatchNumber+1, use_only_cached)
        if batch is None:
            return False
        self.currentBatchNumber += 1
        self.num_loaded_images += len(batch.psds)
        self._calcLogSpectraMeanAndStdDev()
        self._calcMF2DMeanAndStdDev()
        return True
    
    def calculateStatistics(self, use_only_cached=False):
        self.currentBatchNumber = -1
        self.num_loaded_images = 0
        while True:
            batch = self._loadAndCalcImageBatchCachedInfo(self.currentBatchNumber+1, use_only_cached)
            if batch is None:
                break
            self.currentBatchNumber += 1
            self.num_loaded_images += len(batch.psds)
        self._calcLogSpectraMeanAndStdDev()
        self._calcMF2DMeanAndStdDev()
    
    """
    batch_numbers: list of batch numbers or None for all batches
    which_stats: list of strings of which statistics to recalculate
        Possible values:
        'psd': Power spectra and log power spectra.
        'intensityhistogram': Pixel log intensity histograms.
        'mf2d': Minkowski functionals
    """
    def recalculateStatisticsOnBatches(self, batch_numbers, which_stats):
        if batch_numbers is None:
            batch_numbers = np.arange(1000)
        for batch_number in batch_numbers:
            if batch_number in self.image_batches:
                del self.image_batches[batch_number]
            if 'psd' in which_stats:
                if str(batch_number) in self.smcache_spectra:
                    del self.smcache_spectra[str(batch_number)]
                if str(batch_number) in self.smcache_logspectra:
                    del self.smcache_logspectra[str(batch_number)]
            if 'intensityhistogram' in which_stats:
                if str(batch_number) in self.smcache_logintensityhistograms:
                    del self.smcache_logintensityhistograms[str(batch_number)]
            if 'mf2d' in which_stats:
                if str(batch_number) in self.smcache_mf2d:
                    del self.smcache_mf2d[str(batch_number)]
            batch = self._loadAndCalcImageBatchCachedInfo(batch_number)
            if batch is None:
                break
    
    def calcAndGetLogSpectraHistogramsAtKValues(self, kvalues, hist_range, num_hist_bins):
        hists = []
        for k in kvalues:
            hist = np.zeros(num_hist_bins)
            for _, b in self.image_batches.items():
                (h, bin_edges) = np.histogram(b.logpsds[:, k], num_hist_bins, range=hist_range)
                hist += h
            hists.append(hist)
        bins = []
        for n in range(len(bin_edges)-1):
            bin_center = (bin_edges[n] + bin_edges[n+1])/2.0
            bins.append(bin_center)
        return (bins, hists)
    
    def getImageSetName(self):
        return self.imagesetname
    
    def getLoadedImagesCount(self):
        return self.num_loaded_images
    
    def getImage(self, imageNumber):
        image = self.imageset_file.get(str(imageNumber))
        self.imageShape = image.shape
        return np.array(image)
    
    def getImageShape(self):
        try:
            shape = self.imageShape
        except AttributeError:
            image = self.getImage(0)
            shape = image.shape
            self.imageShape = shape
        return shape
    
    def getLogSpectra(self):
        return (np.arange(len(self.logspectra[0])), self.logspectra)
    
    def getLogSpectraMeanAndStdDev(self):
        return (np.arange(len(self.logspectra[0])), self.logspectra_mean, self.logspectra_stddev)
    
    def getLogPixelIntensityHistogram(self):
        hist = np.zeros(self.num_logintensityhistogram_bins)
        for _, b in self.image_batches.items():
            hist += b.logintensityhistogram
        return (self.logintensityhistogram_binedges, hist)

    def get2DMinkowskiFunctionals(self):
        return (self.mf2d_thresholds, self.mf2d_F, self.mf2d_U, self.mf2d_Chi)
    
    def get2DMinkowskiFunctionalsMeanAndStdDev(self):
        return (self.mf2d_thresholds, self.mf2d_F_mean, self.mf2d_F_stddev, self.mf2d_U_mean, self.mf2d_U_stddev, self.mf2d_Chi_mean, self.mf2d_Chi_stddev)
