import sys
import scipy
import pickle
import numpy as np

from numpy import load
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import meet

args = sys.argv
print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

rand_stat = 42
max_epochs = 10000

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

### 00_ToDo: Gather all the 11 base-modalities (leaving out for now the Z-Normalization)
base_modalities_per_subject = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
]


def calculate_hil_features(transformed):
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


def clip_all(a, b, c):
	print(a.shape)
	smalles_length = min(a.shape[1], b.shape[1], c.shape[1]) - 1 # -1 as indizes start at 0 and len starts at 1
	print(smalles_length)
	print(a[:,:smalles_length].shape)
	return a[:,:smalles_length], b[:,:smalles_length], c[:,:smalles_length]


def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
	# load data in format of (channel x epoch_length x number of epochs)
	title_of_run = title % (identifier + 1)
	hfSEP = load(hfsep_path % (identifier + 1))
	noise = load(noise_path % (identifier + 1))

	### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
	# Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
	raw_title = title_of_run + '_raw'

	### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
	# Compute CSP
	csp_title = title_of_run + '_CSP'
	hfSEP_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_0.npy' % (identifier + 1))
	noise_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_0.npy' % (identifier + 1))
	hfSEP_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_1.npy' % (identifier + 1))
	noise_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_1.npy' % (identifier + 1))
	hfSEP_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_2.npy' % (identifier + 1))
	noise_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_2.npy' % (identifier + 1))

	hfSEP_CSP_0, hfSEP_CSP_1, hfSEP_CSP_2 = clip_all(hfSEP_CSP_0, hfSEP_CSP_1, hfSEP_CSP_2)
	noise_CSP_0, noise_CSP_1, noise_CSP_2 = clip_all(noise_CSP_0, noise_CSP_1, noise_CSP_2)
	
	# Compute CCAr
	ccar_title = title_of_run + '_CCAr'
	a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfSEP[:8,:,:])
	ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfSEP[:8,:,:], axes=(0, 0))
	ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], noise[:8,:,:], axes=(0, 0))
	ccar_filt_hfSEP_1 = np.tensordot(a_ccar[:,1], hfSEP[:8,:,:], axes=(0, 0))
	ccar_filt_noise_1 = np.tensordot(a_ccar[:,1], noise[:8,:,:], axes=(0, 0))
	ccar_filt_hfSEP_2 = np.tensordot(a_ccar[:,2], hfSEP[:8,:,:], axes=(0, 0))
	ccar_filt_noise_2 = np.tensordot(a_ccar[:,2], noise[:8,:,:], axes=(0, 0))

	### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
	hil_csp_title = title_of_run + '_CSP_hil'
	hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
	hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
	hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
	hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
	hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
	hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)

	hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
	hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

	hil_ccar_title = title_of_run + '_CCAR_hil'
	hil_extracted_ccar_filt_hfSEP_0 = calculate_hil_features(ccar_filt_hfSEP_0)
	hil_extracted_ccar_filt_noise_0 = calculate_hil_features(ccar_filt_noise_0)
	hil_extracted_ccar_filt_hfSEP_1 = calculate_hil_features(ccar_filt_hfSEP_1)
	hil_extracted_ccar_filt_noise_1 = calculate_hil_features(ccar_filt_noise_1)
	hil_extracted_ccar_filt_hfSEP_2 = calculate_hil_features(ccar_filt_hfSEP_2)
	hil_extracted_ccar_filt_noise_2 = calculate_hil_features(ccar_filt_noise_2)
	hil_extracted_ccar_hfSEP = np.concatenate((hil_extracted_ccar_filt_hfSEP_0, hil_extracted_ccar_filt_hfSEP_1, hil_extracted_ccar_filt_hfSEP_2), axis=0)
	hil_extracted_ccar_noise = np.concatenate((hil_extracted_ccar_filt_noise_0, hil_extracted_ccar_filt_noise_1, hil_extracted_ccar_filt_noise_2), axis=0)

	hfsep_labels = np.ones(len(hfSEP[0,0,:]), dtype=np.int8)
	noise_labels = np.zeros(len(noise[0,0,:]), dtype=np.int8)
	print('amount of labels: %d' % noise_CSP_0.shape[-1])

	# return the datasets in epoch, channel, time_in_channel - fashion
	return [
		[np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
		#   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
		[np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((np.ones(hfSEP_CSP_0.shape[-1], dtype=np.int8), np.zeros(noise_CSP_0.shape[-1], dtype=np.int8)), axis=0), csp_title],
		[np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
		[np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((np.ones(hfSEP_CSP_0.shape[-1], dtype=np.int8), np.zeros(noise_CSP_0.shape[-1], dtype=np.int8)), axis=0), hil_csp_title],
		[np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
	]
    
import matplotlib 
# matplotlib.rcParams['text.usetex'] = True 
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
hfsep_dat, noise_dat, title = base_modalities_per_subject[-1]
workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, 2)

data, labels, run_title = workload[-2]
data_s03 = data[:,:250]
workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, 3)
data, labels, run_title = workload[-2]
data_s04 = data[:,:250]
workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, 7)
data, labels, run_title = workload[-2]
data_s07 = data[:,:250]

######
#####
######

props = dict(boxstyle='square', facecolor='white', alpha=0.7) 
ftsze = 10
font = FontProperties()    
font.set_family('serif')    
font.set_name('Times New Roman')    
font.set_size(ftsze)

ticks = np.arange(125, 3250, 250)   
ticklabels = ['real', 'imaginary', 'amplitude', 'phase', 'real', 'imaginary', 'amplitude', 'phase', 'real', 'imaginary', 'amplitude', 'phase']

color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()
color3 = 'gray'

one_patch = mpatches.Patch(color=color1, label='1st CSP-Filter')
two_patch = mpatches.Patch(color=color2, label='2nd CSP-Filter')
three_patch = mpatches.Patch(color=color3, label='3rd CSP-Filter')

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True) 
axes[0].plot(np.arange(0,1000), data_s03[:1000,45:70], label='S03', linewidth=0.25, color=color1)
axes[0].plot(np.arange(1000, 2000), data_s03[1000:2000,45:70], label='S03', linewidth=0.25, color=color2)
axes[0].plot(np.arange(2000, 3000), data_s03[2000:3000,45:70], label='S03', linewidth=0.25, color=color3)
axes[0].plot(np.arange(0,3000), np.zeros(3000), linewidth=1, color="black") 
axes[1].plot(np.arange(0,1000), data_s04[:1000,45:70], label='S04', linewidth=0.25, color=color1)
axes[1].plot(np.arange(1000, 2000), data_s04[1000:2000,45:70], label='S04', linewidth=0.25, color=color2) 
axes[1].plot(np.arange(2000, 3000), data_s04[2000:3000,45:70], label='S04', linewidth=0.25, color=color3) 
axes[1].plot(np.arange(0,3000), np.zeros(3000), linewidth=1, color="black")

axes[1].plot(np.arange(15,115), np.ones(100) * -3.35, linewidth=1, color="blue") 
plt.text(14, -3.85, '10 ms', fontsize=8, color='blue') 

from matplotlib.patches import ConnectionPatch 

con = ConnectionPatch(xyA=(250, 4), coordsA=axes[0].transData, xyB=(250, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con) 
con1 = ConnectionPatch(xyA=(500, 4), coordsA=axes[0].transData, xyB=(500, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con1)
con2 = ConnectionPatch(xyA=(750, 4), coordsA=axes[0].transData, xyB=(750, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con2)
con3 = ConnectionPatch(xyA=(1000, 4), coordsA=axes[0].transData, xyB=(1000, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con3)

cona = ConnectionPatch(xyA=(1250, 4), coordsA=axes[0].transData, xyB=(1250, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(cona) 
con1a = ConnectionPatch(xyA=(1500, 4), coordsA=axes[0].transData, xyB=(1500, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con1a)
con2a = ConnectionPatch(xyA=(1750, 4), coordsA=axes[0].transData, xyB=(1750, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con2a)
con3a = ConnectionPatch(xyA=(2000, 4), coordsA=axes[0].transData, xyB=(2000, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con3a) 

conb = ConnectionPatch(xyA=(2250, 4), coordsA=axes[0].transData, xyB=(2250, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(conb) 
con1b = ConnectionPatch(xyA=(2500, 4), coordsA=axes[0].transData, xyB=(2500, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con1b)
con2b = ConnectionPatch(xyA=(2750, 4), coordsA=axes[0].transData, xyB=(2750, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con2b)
con3b = ConnectionPatch(xyA=(3000, 4), coordsA=axes[0].transData, xyB=(3000, -4), coordsB=axes[1].transData, color='black', alpha=0.65, linestyle='solid', linewidth=0.5) 
fig.add_artist(con3b)

axes[0].spines['top'].set_visible(False) 
axes[0].spines['bottom'].set_visible(False) 
axes[1].spines['top'].set_visible(False) 

axes[0].axes.xaxis.set_visible(False)
axes[1].set_xticks(ticks=ticks) 
axes[1].set_xticklabels(ticklabels)
# axes[2].tick_params(axis='x', which='minor', pad=15)

plt.legend(handles=[one_patch, two_patch, three_patch],
  fontsize=ftsze, loc='upper center', bbox_to_anchor=(0.495, 2.5), fancybox=True, shadow=True, ncol=3) 
plt.xlabel('12 concatenated feature-types (10 ms to 35 ms post-stimulus each) per example single-trial', fontsize=ftsze, labelpad=15) 
plt.ylabel(r'amplitude (a.u.) / phase (radians from $-\pi$ to $\pi$)', position=(1.0, 1.1), fontproperties=font, fontsize=ftsze, labelpad=15) 
plt.text(10, 13.1, 'S03', fontsize=ftsze) 
plt.text(10, 3.5, 'S04', fontsize=ftsze)
plt.xlim([0,3000]) 
plt.ylim([-4, 4]) 
fig.set_size_inches(13, 5)    
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/scatter_csp_hil_insight_small_new_max_min.png', dpi=300, bbox_inches='tight') 
plt.close('all') 