# imports
import numpy as np
import scipy.linalg
from scipy.signal import hilbert, sosfiltfilt, butter

def pearson_xcorr(curve, pattern):
    """ Computes the Pearson cross-correlation between a pattern and a curve.
        
        It returns a numpy array containing the Pearson correlation values
        between a given pattern and all sub-curves of the same length.
        The pattern is successively shifted from 0 to len(curve)-len(pattern)
        positions. Then, the correlation is computed and stored in the result
        array at the position corresponding to the number of positions the
        pattern has been shifted.
        
        Args
            curve (numpy.array): curve on which we look for the pattern
            pattern: (numpy.array) pattern to be correlated with the curve
            
        Returns
            A numpy.array containing Pearson correlation coefficients
    """
    nb_of_corr = len(curve) - len(pattern) + 1
    pcc  = np.zeros (nb_of_corr)
    for i in range(len(pcc)):
        pcc[i]  = np.corrcoef (pattern, curve[i: i + len(pattern)])[0, 1]
     
    return np.array(pcc)

def fast_pearson_xcorr(curve, pattern, step_size = 100):
    """ Faster Pearson cross-correlation between a pattern and a curve.
        
        Caution: too large patterns will lead to poor performances and
        too large parameters in general may lead to invalid results.
        
        It returns a numpy array containing the Pearson correlation values
        between a given pattern and all sub-curves of the same length.
        The pattern is successively shifted from 0 to len(curve)-len(pattern)
        positions. Then, the correlation is computed and stored in the result
        array at the position corresponding to the number of positions the
        pattern has been shifted.
        
        Args
            curve (numpy.array): curve on which we look for the pattern
            pattern: (numpy.array) pattern to be correlated with the curve
        
        Kwargs
            step_size (int): number of "shifts" simultaneously processed.
                The larger, the faster but the larger the more probable the
                resulting floatting point matrix will have bad properties and
                will lead to invalid results.
            
        Returns
            A numpy.array containing Pearson correlation coefficients
    """
    curve = np.flip(curve)
    plen = len(pattern)
    nb_of_corr = len(curve)- plen + 1
    pcc = []
    
    for step_start in range(0,nb_of_corr,step_size):
        end = min(step_size,nb_of_corr-step_start)
        stacked = scipy.linalg.circulant(curve[step_start:step_start+plen-1+end])[plen-1:,:plen]    
        covar = np.cov (pattern, stacked)
        pcc += [ covar[0,i] / np.sqrt(covar[0,0]*covar[i,i]) for i in range(1,end+1) ]

    pcc.reverse()
    return np.array(pcc)

def simplify_pattern(curve, decimation=256, sos=butter(4, 0.01, output='sos')):
    """Simplify a pattern by applying an envelop detection, a low-pass filter
    and a decimation.
    
        Args
            - curve (numpy.array): the curve to process
    
        Kwargs
            - decimation (int): the decimation paramter (one sample kept among
              `decimation`).
            - sos (numpy.array): signal filter in the `sos` form (refer to
              scipy.signal for more details).
        
        Returns
            A numpy array containing the processed curve (filtered and decimated).
    """
    envelop_curve = np.abs(hilbert(curve))
    filtered_curve = sosfiltfilt(sos, envelop_curve)
    indexes = [ (i%decimation)==0 for i in range(len(filtered_curve)) ]
    decimated_curve = filtered_curve.compress(indexes)
    return decimated_curve

def detect_peaks(correlation, correlation_bis, threshold=0.8, min_gap=200):
    """Detects peaks in a correlation result.
    
       Consecutive high values are omitted if the gap between the index is
       smaller than `min_gap`.
       The `keep` argument can take different values depending on which sample
       should be kept in case more than a single sample reach the threshold
       consecutively.
       
       Args
           - correlation (numpy.array): the correlation result from pattern.
           - correlation_bis (numpy.array): the correlation result from pattern_bis.
       
       Kwargs
           - threshold (float): the threshold above which a peak can be detected.
           - min_gap (int): the minimum distance between two peaks (default to 1).
       
       Returns
           A list of peak abscissas.
       """
    detected_peaks = np.where(correlation>threshold)[0]
    detected_peaks2 = np.where(correlation_bis>threshold)[0]
    if len(detected_peaks) == 0:
        res = []
    else:
        res = [detected_peaks[0]]
        for i in range(1, len(detected_peaks)):
            if detected_peaks[i] - res[-1] < min_gap:
                if detected_peaks[i] > res[-1]:
                    res[-1] = detected_peaks[i]
                continue
            res = np.concatenate((res, detected_peaks[i:i+1]))

    if len(detected_peaks2) == 0:
        res2 = []
    else:
        res2 = [detected_peaks2[0]]
        for i in range(1, len(detected_peaks2)):
            if detected_peaks2[i] - res2[-1] < min_gap:
                if detected_peaks2[i] > res2[-1]:
                    res2[-1] = detected_peaks2[i]
                continue
            res2 = np.concatenate((res2, detected_peaks2[i:i+1]))
    

    return res, res2

def detect_first_peak(pcc,peaks,wrange=20):
    """Compute the first (undetected) peak when he is smaller than the others.

       This function compute the expected position of the first peak (by taking the minimum interval size
       based on the peaks parameter). Then, the maximum correlation is looked for in a window of width 2*wrange.

        Args
            - pcc (numpy.array): cross-correlation vector in which we wish to detect peaks.
            - peaks (list): list of detected peaks.
        
        Kwargs
            - wrange (int): range around the expected peak position to look for higher correlations.
        
        Returns
            
    """
    peaks_np = np.array(peaks)
    min_interval = np.min(peaks_np[1:] - peaks_np[:-1])
    window = np.abs(pcc)[peaks[0]-min_interval-wrange:peaks[0]-min_interval+wrange]
    first_peak = peaks[0]-min_interval-wrange+np.argmax(window)
    return first_peak
