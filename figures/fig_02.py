import sys
import meet  
import numpy as np  
import matplotlib as mpl 
mpl.use('pgf') 
import matplotlib.pyplot as plt  
import scipy
import scipy.stats
from scipy import signal  
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve  
from numpy import save
from meet.spatfilt import CSP

#-#-#import helper_functions 
#-#-#from tqdm import trange 
#-#-#from plot_settings import *

### ### ### ### ###
### Taken from Gunnar's helper script
### ### ### ### ###
####################################
# apply some settings for plotting #
####################################
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.unicode_minus'] = False

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

pgf_with_latex = {
    "pgf.texsystem": "lualatex",     # Use xetex for processing
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r'\usepackage[sc]{mathpazo}',
        r'\usepackage{sfmath}',
        r'\usepackage{xcolor}',     # xcolor for colours
        r'\usepackage[super]{nth}', # nth for counts
        r'\usepackage{textgreek}',
        r'\usepackage{amsmath}',
        r'\usepackage{marvosym}',
        r'\usepackage{graphicx}',
        r'\usepackage{fontspec}'
        r'\setmainfont{Source Sans Pro}'
    ])
}

mpl.rcParams.update(pgf_with_latex)

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False, labelbottom=False)


### ### ### ### ###
### Definition: utility-function / global vars
### ### ### ### ###
offset = 1000
s_rate = 10000
stim_per_sec = 4
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]

def identify_triggers(trigger_signal, estimated_trigger_distance, indicator_value):  
  
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
    ### ToDo: Document and keep in mind the situation of max_until_... 
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

def eigh(cov1, cov2):
    rank = np.linalg.matrix_rank(cov2)
    w, v = np.linalg.eigh(cov2)
    # get whitening matrix
    W = v[:,-rank:]/np.sqrt(w[-rank:])
    cov1_white = W.T.dot(cov1).dot(W)
    eigvals, eigvect = np.linalg.eigh(cov1_white)
    return (
            np.sort(eigvals)[::-1],
            W.dot(eigvect)[:,np.argsort(eigvals)[::-1]])

def bCSTP(data1, data2, num_iter, t_keep, s_keep):
    n_ch, n_dp, n_trials = data1.shape
    t_keep = np.r_[n_dp,
            np.linspace(t_keep, n_dp, num_iter).astype(int)[::-1]]
    s_keep = np.linspace(s_keep, n_ch, num_iter).astype(int)[::-1]
    T_FILT = [np.eye(n_dp)]
    S_FILT = []
    S_EIGVAL = []
    T_EIGVAL = []
    for i in range(num_iter):
        print('bCSTP-iteration num %d' % (i + 1))
        # obtain spatial filter
        temp1 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data1, axes=(0,1))
        temp2 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data2, axes=(0,1))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        S_FILT.append(v)
        S_EIGVAL.append(w)
        # obtain temporal filter
        temp1 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data1, axes=(0,0))
        temp2 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data2, axes=(0,0))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        T_FILT.append(v)
        T_EIGVAL.append(w)
    return S_EIGVAL, T_EIGVAL, S_FILT, T_FILT[1:]

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
### Data loading
### ### ### ### ###
hfSEP_win = [100, 350]
noise_win = [0, 250]
intrplt_win = [-80, 30]

idx = 2
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

triggers_for_kx_combined = identify_triggers(kx_data_combined[8], 300, min(kx_data_combined[8]))

intrplt_kx_data_combined = meet.basic.interpolateEEG(kx_data_combined[:8], triggers_for_kx_combined, intrplt_win)
intrplt_kx_data_combined = np.append(intrplt_kx_data_combined, np.expand_dims(kx_data_combined[8], axis=0), axis=0)

### ### ### ### ###
### Basic preprocessing without outlier-rejection
### ### ### ### ###
intrplt_filt_500_900_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000)
intrplt_filt_500_900_kx = np.append(intrplt_filt_500_900_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)

epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)

### ### ### ### ###
### Advanced Preprocessing on data without outlier-rejection
### ### ### ### ###

# CSP is the signal decomposition using two different signal modalities, due to different points in time, but same preprocessing
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1, order='F'), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1, order='F'))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
csp_filt_epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
csp_filt_epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)

# Define value needed for Gunnar's script
srate = 10000
marker = triggers_for_kx_combined
stim_data_sigma_hilbert = scipy.signal.hilbert(csp_filt_epoched_intrplt_filt_500_900_kx)

### ### ### ### ###
### Continue w. Gunnar's script
### ### ### ### ###
# get the trials
trial_win_ms = [0,80]
trial_win = np.round(np.array(trial_win_ms)/1000.*srate
        ).astype(int)
trial_t = (np.arange(trial_win[0], trial_win[1], 1)/
        float(srate)*1000)

# for high-pass
trials_sigma = meet.epochEEG(stim_data_sigma_hilbert, marker, trial_win)

burst_mask = np.all([trial_t>=10, trial_t<=35], 0)
noise_mask = np.all([trial_t>=45, trial_t<=70], 0)

# remove outlier trials
burst_rms = np.sqrt(np.mean(trials_sigma.real[burst_mask]**2, 0))
burst_rms_q25 = scipy.stats.scoreatpercentile(burst_rms, 25)
burst_rms_q50 = np.median(burst_rms)
burst_rms_q75 = scipy.stats.scoreatpercentile(burst_rms, 75)
burst_iqr = burst_rms_q75 - burst_rms_q25

noise_rms = np.sqrt(np.mean(trials_sigma.real[noise_mask]**2, 0))
noise_rms_q25 = scipy.stats.scoreatpercentile(noise_rms, 25)
noise_rms_q50 = np.median(noise_rms)
noise_rms_q75 = scipy.stats.scoreatpercentile(noise_rms, 75)
noise_iqr = noise_rms_q75 - noise_rms_q25

inlier_trials = np.all([
    burst_rms > (burst_rms_q50 - 1.5 * burst_iqr),
    burst_rms < (burst_rms_q50 + 1.5 * burst_iqr),
    noise_rms > (noise_rms_q50 - 1.5 * noise_iqr),
    noise_rms < (noise_rms_q50 + 1.5 * noise_iqr)],0)

trials_sigma = trials_sigma[:,inlier_trials]

plot_max=80
scatter_cmap_inst = mpl.cm.get_cmap('hsv')
gradient = np.linspace(0,1,256)
gradient = np.vstack([gradient, gradient])

burst_order = np.random.choice(
        np.size(trials_sigma[burst_mask]),
        np.size(trials_sigma[burst_mask]),
        replace=False)
noise_order = np.random.choice(
        np.size(trials_sigma[noise_mask]),
        np.size(trials_sigma[noise_mask]),
        replace=False)

period_length = 1000/600

####################
# plot the results #
####################

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#fig = plt.figure(figsize=(5.51181, 5.5))
fig = plt.figure(figsize=(8.267715, 8.25))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=1,
        figure=fig, height_ratios=(1,3))

burst_ax = fig.add_subplot(gs[0,:], frameon=False)
burst_ax.plot(trial_t, trials_sigma.real.mean(-1), 'k-')
burst_ax.plot([0,0], [0,1], 'k-', transform=burst_ax.transAxes)
burst_ax.plot([0,1], [0,0], 'k-', transform=burst_ax.transAxes)
burst_ax.text(0.01, 0.95, r'$\textbf{band-pass}$' + '\n' + r'(500 Hz -- 900 Hz)',
        ha='left', va='top', multialignment='center',
        transform=burst_ax.transAxes, fontsize=15)
burst_ax.set_xlim([0,80])
burst_ax.set_ylim([-1.5, 1.5])
burst_ax.set_xlabel('time relative to stimulus (ms)')
burst_ax.set_ylabel('ampl. (a.u.)')

burst_ax.axvspan(0,10, color='black', alpha=0.2)
burst_ax.axvspan(10,35, color=color1, alpha=0.4)
burst_ax.axvspan(35,45, color='black', alpha=0.2)
burst_ax.axvspan(45,80, color=color2, alpha=0.4)

with plt.rc_context({"xtick.major.pad": -4, "figure.figsize": [5, 3.75], "figure.subplot.left": 0.155, "figure.subplot.right": 0.805}):
  gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,2,gs[1,:],
          height_ratios=(1,0.05), hspace=0.25, wspace=0.1)
  burst_polar_ax = fig.add_subplot(gs1[0,0], polar=True)
  # TODO#1, figsize=[# TODO: WRITE HERE THE FIGURE SIZES THAT I COULD START OUT WITH, DEFAULT: [6.4, 4.8]])
  # TODO#2: Try moving the labels (0-\pi) to the corner of the radius and move the two plots closer together
  burst_polar_ax.scatter(
          np.angle(trials_sigma[burst_mask]).ravel()[burst_order],
          np.abs(trials_sigma[burst_mask]).ravel()[burst_order],
          c = scatter_cmap_inst(((
              np.ones(trials_sigma[burst_mask].shape)*
              trial_t[burst_mask][:,np.newaxis]).ravel()[burst_order])
              /(1000/600) % 1), alpha=0.6, edgecolors='none', s=10,
          rasterized=True)

  burst_polar_ax.set_title(r'$\textbf{ampl.}$ (a.u.) $\textbf{and phase}$ (rad)' +
          '\n' + r'$\textbf{during burst}$ (10 ms - 35 ms)',
          fontsize=15, multialignment='center', color=color1, pad=20)
  burst_polar_ax.set_xticks(np.linspace(0,2*np.pi,4, endpoint=False))
  burst_polar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
          r'$3\pi /2$'])

  plt.setp(burst_polar_ax.spines.values(), color=color1)
  plt.setp(burst_polar_ax.spines.values(), linewidth=2)

  noise_polar_ax = fig.add_subplot(gs1[0,1], polar=True, sharey =burst_polar_ax)
  noise_polar_ax.scatter(
          np.angle(trials_sigma[noise_mask]).ravel()[noise_order],
          np.abs(trials_sigma[noise_mask]).ravel()[noise_order],
          c = scatter_cmap_inst(((
              np.ones(trials_sigma[noise_mask].shape)*
              trial_t[noise_mask][:,np.newaxis]).ravel()[noise_order])
              /(1000/600) % 1), alpha=0.6, edgecolors='none', s=10,
          rasterized=True)

  noise_polar_ax.set_title(r'$\textbf{ampl.}$ (a.u.) $\textbf{and phase}$ (rad)' + '\n' + r'$\textbf{during noise}$ (rand. 25 ms)', fontsize=15, multialignment='center',
          color=color2, pad=20)
  noise_polar_ax.set_xticks(np.linspace(0,2*np.pi,4, endpoint=False))
  noise_polar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
          r'$3\pi /2$'])

  plt.setp(noise_polar_ax.spines.values(), color=color2)
  plt.setp(noise_polar_ax.spines.values(), linewidth=2)

  # Create offset transform by 2 points in x and 4 in y direction
  label_dx = 2/72.; label_dy = 4/72. 
  label_offset = mpl.transforms.ScaledTranslation(label_dx, label_dy,
          fig.dpi_scale_trans)

  burst_polar_ax.set_rlim([0,4])
  noise_polar_ax.set_rlim([0,4])
  burst_polar_ax.set_rticks([1,2,3])
  noise_polar_ax.set_rticks([1,2,3])
  burst_polar_ax.set_rlabel_position(0)
  noise_polar_ax.set_rlabel_position(0)

  # handling of labels like 20, 40, 60 in PNAS article
  [l.set_linewidth(1) for l in burst_polar_ax.yaxis.get_gridlines()]
  [l.set_color('0.9') for l in burst_polar_ax.yaxis.get_gridlines()]
  [l.set_zorder(10) for l in burst_polar_ax.yaxis.get_gridlines()]
  [l.set_linewidth(1) for l in burst_polar_ax.xaxis.get_gridlines()]
  [l.set_color('0.9') for l in burst_polar_ax.xaxis.get_gridlines()]
  [l.set_zorder(10) for l in burst_polar_ax.xaxis.get_gridlines()]
  [l.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.6,
      boxstyle='circle')) for l in burst_polar_ax.yaxis.get_ticklabels()]
  [l.set_transform(l.get_transform() + label_offset)
          for l in burst_polar_ax.yaxis.get_ticklabels()]

  [l.set_linewidth(1) for l in noise_polar_ax.yaxis.get_gridlines()]
  [l.set_color('0.9') for l in noise_polar_ax.yaxis.get_gridlines()]
  [l.set_zorder(10) for l in noise_polar_ax.yaxis.get_gridlines()]
  [l.set_linewidth(1) for l in noise_polar_ax.xaxis.get_gridlines()]
  [l.set_color('0.9') for l in noise_polar_ax.xaxis.get_gridlines()]
  [l.set_zorder(10) for l in noise_polar_ax.xaxis.get_gridlines()]
  [l.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.6,
      boxstyle='circle')) for l in noise_polar_ax.yaxis.get_ticklabels()]
  [l.set_transform(l.get_transform() + label_offset)
          for l in noise_polar_ax.yaxis.get_ticklabels()]

cbar_ax = fig.add_subplot(gs1[1,:])
cbar_ax.imshow(scatter_cmap_inst(gradient), aspect='auto')
cbar_ax.tick_params(left=False, labelleft=False, right=False, labelright=False,
        top=False, labeltop=False, bottom=True, labelbottom=True)
cbar_ax.set_title(r'$\textbf{phase}$ (rad) $\textbf{of 600 Hz}$', fontsize=15)

cbar_ax.set_xticks(np.linspace(0,256, 5))
cbar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
        r'$3\pi /2$', r'$2\pi$'], fontsize=15)
cbar_ax.set_xlim([0, 256])

gs.tight_layout(fig ,pad=1.0, h_pad=1.5, w_pad=1.5)

fig.canvas.draw()
dx, dy = 0, -2/72.
offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
offset2 = mpl.transforms.ScaledTranslation(0,
        burst_ax.get_xticklabels()[1].get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin -
        burst_ax.get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin,
        fig.dpi_scale_trans)
burst_shift_transform = mpl.transforms.blended_transform_factory(
        burst_ax.transData, burst_ax.transAxes) + offset + offset2

# ToDo: Add helper script, then it should hopefully work ...
burst_ax.text(0, 0, r'\rotatebox[origin=c]{180}{\Lightning}',
        ha='right', va='top', size=16, transform=burst_shift_transform)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# fig.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/Figure_03_k003_newest_new.pdf', dpi=600)
fig.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/Figure_03_k003_newest_new.png', dpi=600)