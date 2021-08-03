import sys
import meet  
import numpy as np  
import matplotlib.pyplot as plt  
import scipy
from scipy import signal  
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve  
from numpy import save
from meet.spatfilt import CSP


### ### ### ### ###
### Definition: utility-function / global vars
### ### ### ### ###
offset = 1000
s_rate = 10000
stim_per_sec = 4
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]


def identify_triggers(trigger_signal, estimated_trigger_distance, indicator_value):  
    # read the trigger points used in the original data and return these
    # 1st version: define the timestamp when the signal is at zero again as "start of trigger"  
    triggers = [0]  
    ttl_found = False  
    ttl_samples_ctr = 0  
      
    for idx, data_point in enumerate(trigger_signal):  
        if triggers[-1] + int(0.9 * estimated_trigger_distance) <= idx and trigger_signal[idx] == indicator_value:  
            ttl_found = True  
            ttl_samples_ctr = ttl_samples_ctr + 1  
        else:  
            ttl_found = False  
        if ttl_samples_ctr > 0 and not ttl_found:  
            triggers.append(idx + 40) # -1 as to change of index for old position; -41 as to offset-correciton  
            ttl_samples_ctr = 0  
          
    return triggers[1:] 


def trim_data_to_contain_only_valid_seconds(data_to_trim, triggers_of_data_to_trim):
  """
  General Idea:
  In the underlying data the subjects were stimulated by 4Hz.
  The window around a stimulus that we're interested in is 150ms long: 
  [<<-55ms Noise? -10ms>> <<-8ms Intrplt 2ms>> <<5ms hfSEP 45ms>> <<45ms Noise? 95ms>>]:= 150ms
  This leaves us with ~50ms of unused data around each end of our window of interest. (1s / 4)
  If we define the offset to be so that we place the start of data existing in the non-used
  window our seconds-segments will always include the whole window we are interested in, 
  even after outlier-rejection and then re-calculating the trigger-points.
  We only have to pay attention and perform this step everytime first, meaning:
  after filtering but prior to combining the data of different measurements per subject together
  
  Attention:
  Makes use of global variables offset and s_rate!

  Sanity-Checks:
  Check if the length of data returned can be divided by s_rate w.o. rest.
  Or plot some parts of the data, including the trigger-channel in steps of s_rate and see triggers overlaid
  """

  if (triggers_of_data_to_trim[0] - offset) > 0:
    timestamp_last_trigger_to_use_safely = triggers_of_data_to_trim[np.arange(0, len(triggers_of_data_to_trim), 4)[-1]]
    return data_to_trim[:,triggers_of_data_to_trim[0] - offset : timestamp_last_trigger_to_use_safely - offset]  
  else:
    timestamp_last_trigger_to_use_safely = triggers_of_data_to_trim[np.arange(1, len(triggers_of_data_to_trim), 4)[-1]]
    return data_to_trim[:,triggers_of_data_to_trim[1] - offset : timestamp_last_trigger_to_use_safely - offset]


def get_indices_for_noise(triggers_to_get_indices_for):
  """
  General Idea:
  If we defined a window of 'safety', which we have to randomly place the noise-window in,
  then we can safely use random noise-window placements in this window.
  The window around a stimulus that we're interested in is 150ms long: 
  [<<-55ms Noise? -10ms>> <<-8ms Intrplt 2ms>> <<5ms hfSEP 45ms>> <<45ms Noise? 95ms>>]:= 150ms
  --> Follows the noise-window can start in [-55ms to -50ms] and [45ms to 55ms].
  """
  possible_neg_noise_starting_indices = np.arange(-550, -500, 1)
  possible_pos_starting_indices = np.arange(450, 550)
  possible_noise_indices_per_sample = np.append(possible_neg_noise_starting_indices, possible_pos_starting_indices)
  return np.asarray(triggers_to_get_indices_for) + np.random.choice(possible_noise_indices_per_sample, len(triggers_to_get_indices_for))


def normalize_min_max(to_be_normalized): 
    return (to_be_normalized - np.min(to_be_normalized)) / (np.max(to_be_normalized) - np.min(to_be_normalized)) 


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized)) 


def reject_outlier_seconds_using_lsd(dat, chan=None, measure='ED', thresh=0.13, plot_eds=False, srate=10000, identifier=-1): 
  """
  Compute outliers as in waterstraat_noninvasive_2016, however with extended functionality
  -> doi: 10.1016/j.clinph.2015.12.005 
  """
    lsd_baseline = [] 
    euclidean_distances = [] 
    num_secs = int(len(dat[0]) / srate)
 
    if chan is None: 
        pxx_per_sec = np.asarray([scipy.signal.welch(dat[:8, srate * i : srate * (i + 1)], fs=srate)[1] for i in range(num_secs)]) 
        lsd_baseline = np.median(pxx_per_sec, axis=0).mean(0) 
        euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec.mean(1))**2).sum(1)) # 0_==sec; 1_==freq
    else: 
        pxx_per_sec = np.asarray([scipy.signal.welch(dat[chan, srate * i : srate * (i + 1)], fs=srate)[1] for i in range(num_secs)])  
        lsd_baseline = np.median(pxx_per_sec, axis=0).mean(0) 
        euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec)**2).sum(1)) 
     
    if plot_eds: 
        import matplotlib.pyplot as plt
        plt.plot(euclidean_distances, color='blue', label='ED in channel FZ') 
        plt.plot(np.ones(len(euclidean_distances)) * (0.25 * 3), color='yellow', linestyle='-.', label='Threshold at %f' % (0.25 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.2 * 3), color='black', linestyle=':', label='Threshold at %f' % (0.2 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.15 * 3), color='yellow', linestyle='--', label='Threshold at %f' % (0.15 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.1 * 3), color='black', linestyle='--', label='Threshold at %f' % (0.1 * 3)) 
        plt.title('ED-Plot to visually derive the threshold for outlier rejection of K00%d' % identifier) 
        plt.xlabel('Seconds in combined data of K00%d; intrplt. stim. 1, 2, 3 comb.; Zoomed in' % identifier) 
        plt.ylabel('Euclidean Distance') 
        plt.ylim([0.0, 3.0]) 
        plt.legend() 
        plt.savefig('/media/christoph/Volume/Masterthesis/Presentations/Zwischenpräsentation_18_09/k00%d_ED_Plot_FZ_Zoomed_In' % identifier) 
        plt.clf()
     
    ed_mean = euclidean_distances.mean() 
    ed_std = euclidean_distances.std() 
    
    lsds_kept_within_ed_threshold = 0 
    lsds_kept_one_sd_over_mean = 0 
    lsds_kept_two_sd_over_mean = 0 

    if chan is None: 
       lsds_kept_within_ed_threshold = np.arange(0, len(euclidean_distances), 1)[euclidean_distances <= out_rej_thresh_mean[identifier]] 
       lsds_kept_one_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - ed_std) <= 0] 
       lsds_kept_two_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - (2*ed_std) <= 0)] 
    else:
       lsds_kept_within_ed_threshold = np.arange(0, len(euclidean_distances), 1)[euclidean_distances <= out_rej_thresh_fz[identifier]] 
       lsds_kept_one_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - ed_std) <= 0] 
       lsds_kept_two_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - (2*ed_std) <= 0)]
     
    secs_to_keep = [] 
    if measure is 'ED': 
        secs_to_keep = lsds_kept_within_ed_threshold 
    elif measure is 'ED1SD': 
        secs_to_keep = lsds_kept_one_sd_over_mean 
    elif measure is 'ED2SD': 
        secs_to_keep = lsds_kept_two_sd_over_mean 
    else: 
        raise ValueError('Unknown measure %s specified; Must be one of: {ED, ED1SD, ED2SD}' % measure) 
         
    conc_kept_data_ids = np.concatenate([np.full(srate, True) if i in secs_to_keep else np.full(srate, False) for i in range(num_secs)]) 
    return dat[:,conc_kept_data_ids]


chan_names = ['FZ', 'F3', 'FC5', 'CZ', 'C3', 'CP5', 'T7', 'CP1']

data_for_visualization = [
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/02-K01_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/03-K01_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/04-K01_stim3.dat'],
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/02-K02_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/03-K02_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/04-K02_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/02-K03_stim1_2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/03-K03_stim2_2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/04-K03_stim3_2.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/02-K04_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/03-K04_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/04-K04_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/02-K05_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/03-K05_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/04-K05_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/02-K06_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/03-K06_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/04-K06_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/02-K07_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/03-K07_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/04-K07_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/02-K08_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/03-K08_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/04-K08_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/02-K09_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/03-K09_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/04-K09_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/02-K10_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/03-K10_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/04-K10_stim3.dat'] 
]


### ### ### ### ###
### Data loading ## <-- The script is called with a parameter read into the idx variable, that's how you iterate over subjects
### ### ### ### ###
hfSEP_win = [50, 450]
noise_win = [-500, -100]
intrplt_win = [-80, 30]

idx = int(sys.argv[1])
print('Running final data prepping for k00%d ...' % (idx + 1))
data_for_train = data_for_visualization[idx]  

data_kx_stim1 = meet.readBinary(data_for_train[0], num_channels=9, data_type='float8')
data_kx_stim2 = meet.readBinary(data_for_train[1], num_channels=9, data_type='float8') 

# K002, stim3: Channel-Drift, daher nur 10000:1860000!
if idx == 1:
  data_kx_stim3 = meet.readBinary(data_for_train[2], num_channels=9, data_type='float8')[:,10000:1860000]
else:  
  data_kx_stim3 = meet.readBinary(data_for_train[2], num_channels=9, data_type='float8')

triggers_for_data_kx_stim1 = identify_triggers(data_kx_stim1[8], 300, min(data_kx_stim1[8]))
triggers_for_data_kx_stim2 = identify_triggers(data_kx_stim2[8], 300, min(data_kx_stim2[8]))
triggers_for_data_kx_stim3 = identify_triggers(data_kx_stim3[8], 300, min(data_kx_stim3[8]))

trimmed_data_kx_stim1 = trim_data_to_contain_only_valid_seconds(data_kx_stim1, triggers_for_data_kx_stim1)
trimmed_data_kx_stim2 = trim_data_to_contain_only_valid_seconds(data_kx_stim2, triggers_for_data_kx_stim2)
trimmed_data_kx_stim3 = trim_data_to_contain_only_valid_seconds(data_kx_stim3, triggers_for_data_kx_stim3)

### ### ### ### ###
### Basic preprocessing for all the datasets needed
### ### ### ### ###  
kx_data_combined = np.append(trimmed_data_kx_stim1, trimmed_data_kx_stim2, axis=1)  
kx_data_combined = np.append(kx_data_combined, trimmed_data_kx_stim3, axis=1)
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined' % (idx + 1), kx_data_combined)

triggers_for_kx_combined = identify_triggers(kx_data_combined[8], 300, min(kx_data_combined[8]))
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined' % (idx + 1), triggers_for_kx_combined)

intrplt_kx_data_combined = meet.basic.interpolateEEG(kx_data_combined[:8], triggers_for_kx_combined, intrplt_win)
intrplt_kx_data_combined = np.append(intrplt_kx_data_combined, np.expand_dims(kx_data_combined[8], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined' % (idx + 1), intrplt_kx_data_combined)


def prep_without_outlier_rejection():

  ### ### ### ### ###
  ### Basic preprocessing without outlier-rejection
  ### ### ### ### ###
  intrplt_filt_under_100_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=110, s_rate=10000)
  intrplt_filt_under_100_kx = np.append(intrplt_filt_under_100_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx' % (idx + 1), intrplt_filt_under_100_kx)

  intrplt_filt_over_100_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=90, s_rate=10000)
  intrplt_filt_over_100_kx = np.append(intrplt_filt_over_100_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx' % (idx + 1), intrplt_filt_over_100_kx)

  intrplt_filt_over_400_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=400, fs=360, s_rate=10000)
  intrplt_filt_over_400_kx = np.append(intrplt_filt_over_400_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx' % (idx + 1), intrplt_filt_over_400_kx)

  intrplt_filt_500_900_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000)
  intrplt_filt_500_900_kx = np.append(intrplt_filt_500_900_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx' % (idx + 1), intrplt_filt_500_900_kx)

  epoched_kx_data_combined_hfsep = meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  epoched_kx_data_combined_noise = meet.epochEEG(kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  print(epoched_kx_data_combined_hfsep.shape)
  print(epoched_kx_data_combined_noise.shape)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep' % (idx + 1), epoched_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise' % (idx + 1), epoched_kx_data_combined_noise)

  epoched_intrplt_kx_data_combined_hfsep = meet.epochEEG(intrplt_kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_kx_data_combined_noise = meet.epochEEG(intrplt_kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise' % (idx + 1), epoched_intrplt_kx_data_combined_noise)

  epoched_intrplt_filt_under_100_kx_hfsep = meet.epochEEG(intrplt_filt_under_100_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_under_100_kx_noise = meet.epochEEG(intrplt_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), epoched_intrplt_filt_under_100_kx_noise)

  epoched_intrplt_filt_over_100_kx_hfsep = meet.epochEEG(intrplt_filt_over_100_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_over_100_kx_noise = meet.epochEEG(intrplt_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), epoched_intrplt_filt_over_100_kx_noise)

  epoched_intrplt_filt_over_400_kx_hfsep = meet.epochEEG(intrplt_filt_over_400_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_over_400_kx_noise = meet.epochEEG(intrplt_filt_over_400_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), epoched_intrplt_filt_over_400_kx_noise)

  epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d basic preprocessing without outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Z-Normalization of Basic Preprocessing without outlier-rejection
  ### ### ### ### ###

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_kx_data_combined' % (idx + 1), normalize_z(intrplt_kx_data_combined))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_under_100_kx' % (idx + 1), normalize_z(intrplt_filt_under_100_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_over_100_kx' % (idx + 1), normalize_z(intrplt_filt_over_100_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_over_400_kx' % (idx + 1), normalize_z(intrplt_filt_over_400_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_500_900_kx' % (idx + 1), normalize_z(intrplt_filt_500_900_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_kx_data_combined_hfsep' % (idx + 1), normalize_z(epoched_kx_data_combined_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_kx_data_combined_noise' % (idx + 1), normalize_z(epoched_kx_data_combined_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_kx_data_combined_noise' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_noise))

  print('For k00%d z-normalization of basic preprocessing without outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Advanced Preprocessing on data without outlier-rejection
  ### ### ### ### ###

  # CSP is the signal decomposition using two different signal modalities, due to different points in time, but same preprocessing
  # csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8,50:150].reshape(8, -1, order='F'), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1, order='F'))
  csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1, order='F'), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1, order='F'))
  csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)

  hfSEP_CSP_1 = meet.epochEEG(np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,1].T, intrplt_filt_500_900_kx[:8], axes=(0, 0)), triggers_for_kx_combined, hfSEP_win)
  noise_CSP_1 = meet.epochEEG(np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,1].T, intrplt_filt_500_900_kx[:8], axes=(0, 0)), get_indices_for_noise(triggers_for_kx_combined), noise_win)
  hfSEP_CSP_2 = meet.epochEEG(np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,2].T, intrplt_filt_500_900_kx[:8], axes=(0, 0)), triggers_for_kx_combined, hfSEP_win)
  noise_CSP_2 = meet.epochEEG(np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,2].T, intrplt_filt_500_900_kx[:8], axes=(0, 0)), get_indices_for_noise(triggers_for_kx_combined), noise_win)

  # Had to store CSP num 2 correctly, not only twice num 1
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_0' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_0' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_noise)
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_1' % (idx + 1), hfSEP_CSP_1)
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_1' % (idx + 1), noise_CSP_1)
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_2' % (idx + 1), hfSEP_CSP_2)
  save('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_2' % (idx + 1), noise_CSP_2)

  print('For k00%d CSP without outlier-rejection made' % (idx + 1))

  # CCAr is the technique that tries to derive filters that 'modify' the single-trial to be more similar to the single-trial averages
  a_epoched_kx_data_combined_hfsep, b_epoched_kx_data_combined_hfsep, s_epoched_kx_data_combined_hfsep = meet.spatfilt.CCAvReg(epoched_kx_data_combined_hfsep[:8,:,:])
  a_epoched_intrplt_kx_data_combined_hfsep, b_epoched_intrplt_kx_data_combined_hfsep, s_epoched_intrplt_kx_data_combined_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_kx_data_combined_hfsep[:8,:,:])
  a_epoched_intrplt_filt_under_100_kx_hfsep, b_epoched_intrplt_filt_under_100_kx_hfsep, s_epoched_intrplt_filt_under_100_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_under_100_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_over_100_kx_hfsep, b_epoched_intrplt_filt_over_100_kx_hfsep, s_epoched_intrplt_filt_over_100_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_100_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_over_400_kx_hfsep, b_epoched_intrplt_filt_over_400_kx_hfsep, s_epoched_intrplt_filt_over_400_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_400_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])

  ccar_filt_epoched_kx_data_combined_hfsep = np.tensordot(a_epoched_kx_data_combined_hfsep[:,0], epoched_kx_data_combined_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_kx_data_combined_noise = np.tensordot(a_epoched_kx_data_combined_hfsep[:,0], epoched_kx_data_combined_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_kx_data_combined_hfsep = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep[:,0], epoched_intrplt_kx_data_combined_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_kx_data_combined_noise = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep[:,0], epoched_intrplt_kx_data_combined_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep[:,0], epoched_intrplt_filt_under_100_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_noise = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep[:,0], epoched_intrplt_filt_under_100_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep[:,0], epoched_intrplt_filt_over_100_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_noise = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep[:,0], epoched_intrplt_filt_over_100_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep[:,0], epoched_intrplt_filt_over_400_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_noise = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep[:,0], epoched_intrplt_filt_over_400_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_noise = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], epoched_intrplt_filt_500_900_kx_noise[:8,:,:], axes=(0, 0))

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_kx_data_combined_hfsep' % (idx + 1), ccar_filt_epoched_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_kx_data_combined_noise' % (idx + 1), ccar_filt_epoched_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_noise' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d CCAr without outlier-rejection made' % (idx + 1))


### ### ### ### ###
prep_without_outlier_rejection() # outlier-rejection was not included in the work published

### ### ### ### ###
### End of Data Preprocessing Script
### ### ### ### ###
