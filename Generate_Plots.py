import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
from Generate_Stats import StatMaker

class PlotMaker:
    def __init__(self):
        self.colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    """
    Color sequence to use for plotting StatMakers.
    
    colorSequence: list of matplotlib color names
    
    The default color sequence is the same as the matplotlib default color sequence (['C0', 'C1', ... 'C9'])
    """
    def setColorSequence(self, colorSequence):
        self.colors = colorSequence
    
    """
    For all plotting methods:

    statmakers: A list of StatMaker instances to make plots from.
                getImageSetName() is used to get the name to plot for that StatMaker.
    
    All plotting methods plot into the current figure.
    """

    """
    Plots log-log histograms of pixel intensities.
    
    Does a KS test comparing the first histogram to all subsequent ones and returns the results (D, p) as tuples in a list.
    """
    def plotIntensityHistograms(self, statmakers, kstests=False, show_first_kstest_pvalue=False):
        ksret = []
        setnames = []
        for n, sm in enumerate(statmakers):
            num_images = sm.getLoadedImagesCount()
            (binedges, hist) = sm.getLogPixelIntensityHistogram()
            shape = sm.getImageShape()
            hist = hist / float(num_images*shape[0]*shape[1])
            if kstests:
                if n == 0:
                    firstHist = hist
                else:
                    (D, p) = stats.ks_2samp(firstHist, hist)
                    ksret.append((D, p))
            bins = np.zeros(len(binedges)-1)
            for i in range(len(binedges)-1):
                bins[i] = (binedges[i]+binedges[i+1])/2.0
            plt.plot(bins, hist, self.colors[n]+'o')
            setnames.append(sm.getImageSetName())
        bin_width = abs(bins[0] - bins[1])
        titlestr = 'Histogram of Pixel Intensities (bin width '+'{:.2f}'.format(bin_width)+')'
        if kstests and show_first_kstest_pvalue:
            titlestr += ', KS test $p$-value '+'{:.2f}'.format(ksret[0][1])
        plt.title(titlestr)
        plt.legend(setnames)
        plt.xlabel('log($\mu K$)')
        plt.ylabel('Fraction of Pixels in Set')
        plt.yscale('log')
        if kstests:
            return ksret

    """
    Plots the 3 2D Minkowski functions' mean and std. dev.
    """
    def plotMinkowskiFunctionals(self, statmakers):
        plt.suptitle('Minkowski Functions of Log-Norm Images (Mean +/- Std. Dev.)')
        for n, sm in enumerate(statmakers):
            (thresholds, F_mean, F_stddev, U_mean, U_stddev, Chi_mean, Chi_stddev) = sm.get2DMinkowskiFunctionalsMeanAndStdDev()
            c = self.colors[n]
            plt.subplot(131)
            plt.plot(thresholds, F_mean, c+'--', label=sm.getImageSetName())
            plt.plot(thresholds, F_mean+F_stddev, c+'-')
            plt.plot(thresholds, F_mean-F_stddev, c+'-')
            plt.legend()
            plt.subplot(132)
            plt.plot(thresholds, U_mean, c+'--', label=sm.getImageSetName())
            plt.plot(thresholds, U_mean+U_stddev, c+'-')
            plt.plot(thresholds, U_mean-U_stddev, c+'-')
            plt.subplot(133)
            plt.plot(thresholds, Chi_mean, c+'--', label=sm.getImageSetName())
            plt.plot(thresholds, Chi_mean+Chi_stddev, c+'-')
            plt.plot(thresholds, Chi_mean-Chi_stddev, c+'-')
        subplot_titles = ['$V_0$', '$V_1$', '$V_2$']
        subplot_ylabels = ['Area', 'Boundary length', 'Euler characteristic']
        for n in range(3):
            plt.subplot(131+n)
            plt.title(subplot_titles[n])
            plt.xlabel('Threshold')
            plt.ylabel(subplot_ylabels[n])
            #plt.yscale('log')
            
    """
    Runs the KS tests for the 2D Minkowski functionals at each threshold between the first StatMaker and all subsequent StatMakers.
    Returns a tuple (mf_thresholds, kstest_for_V0, kstest_for_V1, kstest_for_V2)
    mf_thresholds are the Minkowski functional thresholds
    kstest_for_Vn is a list that can be indexed as:
    (D, p) = list[statmaker index - 1][threshold index]
    """
    def kstestMinkowskiFunctionals(self, statmakers):
        ret_ksF = []
        ret_ksU = []
        ret_ksChi = []
        for n, sm in enumerate(statmakers):
            (thresholds, F, U, Chi) = sm.get2DMinkowskiFunctionals()
            if n == 0:
                first_F = F
                first_U = U
                first_Chi = Chi
                continue
            ksF = []
            ksU = []
            ksChi = []
            for m in range(len(thresholds)):
                (D, p) = stats.ks_2samp(first_F[:, m], F[:, m])
                ksF.append((D, p))
                (D, p) = stats.ks_2samp(first_U[:, m], U[:, m])
                ksU.append((D, p))
                (D, p) = stats.ks_2samp(first_Chi[:, m], Chi[:, m])
                ksChi.append((D, p))
            ret_ksF.append(ksF)
            ret_ksU.append(ksU)
            ret_ksChi.append(ksChi)
        return (thresholds, ret_ksF, ret_ksU, ret_ksChi)
    
    """
    Plots the KS tests' p-value for the 2D Minkowski functionals at each threshold.
    The first statmaker is compared to all subsequent statmakers.
    (mf_thresholds, ksF, ksU, ksChi) can be gotten from calling kstestMinkowskiFunctionals
    """
    def plotKSTestpMinkowskiFunctionals(self, statmakers, mf_thresholds, ksF, ksU, ksChi):
        firstSMname = statmakers[0].getImageSetName()
        plt.suptitle('KS test $p$-values of Minkowski Functions')
        for n, sm in enumerate(statmakers[1:]):
            setname = sm.getImageSetName()
            c = self.colors[n]
            labelstr = firstSMname+' v. '+setname
            plt.subplot(131)
            plt.plot(mf_thresholds, [Dp[1] for Dp in ksF[n]], c+'-', label=labelstr)
            plt.subplot(132)
            plt.plot(mf_thresholds, [Dp[1] for Dp in ksU[n]], c+'-', label=labelstr)
            plt.subplot(133)
            plt.plot(mf_thresholds, [Dp[1] for Dp in ksChi[n]], c+'-', label=labelstr)
        ax = plt.subplot(131)
        plt.title('KS test of $V_0$')
        plt.xlabel('Threshold')
        plt.ylabel('$p$')
        ax.set_ylim(0.0, 1.0)
        ax.set_ybound(-0.1, 1.1)
        plt.legend()
        ax = plt.subplot(132)
        plt.title('KS test of $V_1$')
        plt.xlabel('Threshold')
        ax.set_ylim(0.0, 1.0)
        ax.set_ybound(-0.1, 1.1)
        ax = plt.subplot(133)
        plt.title('KS test of $V_2$')
        plt.xlabel('Threshold')
        ax.set_ylim(0.0, 1.0)
        ax.set_ybound(-0.1, 1.1)
    
    """
    Plots the log power spectra's mean and std. dev.
    """
    def plotLogPowerSpectra(self, statmakers, k_to_ell_conversion_factor):
        for n, sm in enumerate(statmakers):
            (k, spectra_mean, spectra_stddev) = sm.getLogSpectraMeanAndStdDev()
            ells = k*k_to_ell_conversion_factor
            c = self.colors[n]
            plt.plot(ells, spectra_mean, c+'--', label=sm.getImageSetName())
            plt.plot(ells, spectra_mean+spectra_stddev, c+'-')
            plt.plot(ells, spectra_mean-spectra_stddev, c+'-')
        plt.title('Log Power Spectra (Mean +/- Std. Dev.)')
        plt.legend()
        plt.xlabel(r'$l$')
        plt.xscale('log')
        plt.ylabel('log($\mu K^2$)')
        #locs, labels = plt.yticks()
        #plt.yticks(locs, [(r'$10^{'+str(int(y))+r'}$') for y in locs])
    
    """
    Plots the log power spectra's mean and std. dev from realsm
    Plots the log power spectra for images given by image_idxs from fakesm
    """
    def plotIndividualLogPowerSpectra(self, realsm, fakesm, image_idxs, k_to_ell_conversion_factor):
        c = self.colors[1]
        label = fakesm.getImageSetName()+" ("+str(len(image_idxs))+" images)"
        (k, spectra) = fakesm.getLogSpectra()
        ells = k*k_to_ell_conversion_factor
        for idx in image_idxs:
            plt.plot(ells, spectra[idx], c+'-', label=label)
            label=None

        (k, spectra_mean, spectra_stddev) = realsm.getLogSpectraMeanAndStdDev()
        ells = k*k_to_ell_conversion_factor
        c = self.colors[0]
        plt.plot(ells, spectra_mean, c+'--', label=realsm.getImageSetName()+" mean")
        plt.plot(ells, spectra_mean+spectra_stddev, c+'-', label=realsm.getImageSetName()+" +/- std. dev.")
        plt.plot(ells, spectra_mean-spectra_stddev, c+'-')

        plt.title('Log Power Spectra')
        plt.legend()
        plt.xlabel(r'$l$')
        plt.xscale('log')
        plt.ylabel('log($\mu K^2$)')
        #locs, labels = plt.yticks()
        #plt.yticks(locs, [(r'$10^{'+str(int(y))+r'}$') for y in locs])

    """
    Plots the histograms of the power spectra at specified k's
    
    If kstests is True, also runs the kstest between the histograms of the first statmaker and all subsequent statmakers and returns the results in a list which can be indexed as follows:
    (D, p) = list[statmaker index - 1][k-value index]
    """
    def plotLogPowerSpectraHistogramsAtKValues(self, statmakers, kvalues, k_to_ell_conversion_factor, hist_range, hist_bin_count, kstests=False, show_first_kstest_pvalue=False):
        num_images = []
        hist_bins_list = []
        hists_list = []
        setnames = []
        ksret = []
        for n, sm in enumerate(statmakers):
            num_images.append(sm.getLoadedImagesCount())
            (hist_bins, hists) = sm.calcAndGetLogSpectraHistogramsAtKValues(kvalues, hist_range, hist_bin_count)
            hist_bins_list.append(hist_bins)
            hists_list.append(hists)
            setnames.append(sm.getImageSetName())
            if kstests:
                if n == 0:
                    firstHists = hists
                else:
                    ksret.append([])
        for n, k in enumerate(kvalues):
            plt.subplot(131+n)
            for m, setname in enumerate(setnames):
                hist_bins = hist_bins_list[m]
                hists = hists_list[m]
                hist = hists[n] / num_images[m]
                bin_width = abs(hist_bins[2]-hist_bins[1])
                plt.plot(hist_bins, hist, self.colors[m]+'o-', label=setname)
                if kstests and (m > 0):
                    (D, p) = stats.ks_2samp(firstHists[n] / num_images[0], hist)
                    ksret[m-1].append((D, p))
            titlestr = '$l={'+str(int(k*k_to_ell_conversion_factor))+'}$'
            if kstests and show_first_kstest_pvalue:
                titlestr += ', KS test $p$-value '+'{:.2f}'.format(ksret[0][n][1])
            plt.title(titlestr)
            plt.yscale('log')
            locs, labels = plt.xticks()
            plt.xticks(locs, [(r'$10^{'+str(int(x))+r'}$') for x in locs])
            plt.xlabel('$\mu K^2$')
            if n == 0:
                plt.ylabel('Fraction of Images in Set')
                plt.legend()
        plt.suptitle('Histograms of Power Spectra at given $l$-values (bin width is '+'{:.2f}'.format(bin_width)+' in log-space)')
        if kstests:
            return ksret

    """
    Plot pixel intensity histogram from a StatMaker with a pixel intensity histogram w/ error bars from a file.
    The second histogram comes from a file which was generated using scripts/gen_GAN_intensity_histograms.py
    histsfile_distributionname is the name to use for the plot generated from histsfile
    """
    def plotIntensityHistogramsWithErrorBars(self, statmaker, histsfile, histsfile_distributionname):
        histsf = h5py.File(histsfile, 'r')
        img_width = histsf['imagesize'][0]
        img_height = histsf['imagesize'][1]
        num_hists = histsf['datasize'][0]
        num_images = histsf['datasize'][1]
        bin_edges = histsf['binedges'][:]
        bin_width = bin_edges[2]-bin_edges[1]
        num_pixels = img_width*img_height*num_images
        hists = np.zeros((num_hists, len(bin_edges)-1))
        for n in range(num_hists):
            hists[n, :] = histsf[str(n)][:]
        histsf.close()
        
        (sm_bin_edges, smhist) = statmaker.getLogPixelIntensityHistogram()
        x = np.zeros(len(sm_bin_edges)-1)
        for n in range(len(sm_bin_edges)-1):
            x[n] = (sm_bin_edges[n]+sm_bin_edges[n+1])/2.0
        sm_num_images = statmaker.getLoadedImagesCount()
        sm_shape = statmaker.getImageShape()
        plt.plot(x, np.log10(smhist / float(sm_num_images*sm_shape[0]*sm_shape[1])), self.colors[0]+'o')
        #plt.plot(x, smhist / float(sm_num_images*sm_shape[0]*sm_shape[1]), self.colors[0]+'o')
        
        hist_avg = np.mean(hists, 0)
        x = np.zeros(len(bin_edges)-1)
        for n in range(len(bin_edges)-1):
            x[n] = (bin_edges[n]+bin_edges[n+1])/2.0
        #y = hist_avg / float(num_pixels)
        #yerr = np.sqrt(hist_avg) / float(num_pixels)
        y = np.log10(hist_avg / float(num_pixels))
        # See https://faculty.washington.edu/stuve/log_error.pdf for error bars on a log scale
        # Error = d[hist_avg] = sqrt(hist_avg), d[x] denotes delta-x or uncertainty in x
        # After scaling by 1/num_pixels: d[hist_avg / num_pixels] = d[hist_avg] / num_pixels = sqrt(hist_avg) / num_pixels
        # After log10:
        #    d[log10(hist_avg / num_pixels)] = 
        #    d[log10(hist_avg) - log10(num_pixels)] =
        #    d[log10(hist_avg)] - d[log10(num_pixels)] =
        #    d[log10(hist_avg)] =
        #    d[hist_avg] / (hist_avg * ln(10)) =
        #    sqrt(hist_avg) / (hist_avg * ln(10)) =
        #    1 / (sqrt(hist_avg) * ln(10))
        yerr = 1.0 / (np.sqrt(hist_avg) * np.log(10.0))
        plt.errorbar(x, y, yerr=yerr, fmt=self.colors[1]+'o')
        
        plt.legend([statmaker.getImageSetName(), histsfile_distributionname])
        titlestr = 'Histogram of Pixel Intensities (bin width '+'{:.2f}'.format(bin_width)+')'
        plt.title(titlestr)
        plt.xlabel('log($\mu K$)')
        plt.ylabel('Fraction of Pixels in Set')
        locs, labels = plt.yticks()
        labels = [(r'$10^{'+str(int(y))+r'}$') for y in locs]
        labels[0] = ''
        labels[-1] = ''
        plt.yticks(locs, labels)
