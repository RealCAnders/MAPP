import sys
import scipy
import pickle
import numpy as np

from numpy import load
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import meet

args = sys.argv
print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

rand_stat = 42


base_modalities_per_subject = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
]


def calculate_hil_features(transformed):
  # For the transformed data, calclulate respective Hilbert Transform and return
  # real, imaginary, absolute, and angle thereof
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


def clip_all(a, b, c):
  # clip the data to be of same length
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

   raw_title = title_of_run + '_raw'

   # Compute CSP
   csp_title = title_of_run + '_CSP'
   hfSEP_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_0.npy' % (identifier + 1))
   noise_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_0.npy' % (identifier + 1))
   hfSEP_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_1.npy' % (identifier + 1))
   noise_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_1.npy' % (identifier + 1))
   hfSEP_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_2.npy' % (identifier + 1))
   noise_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_2.npy' % (identifier + 1))

   # Compute CCAr
   ccar_title = title_of_run + '_CCAr'
   a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfSEP[:8,:,:])
   ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], noise[:8,:,:], axes=(0, 0))
   ccar_filt_hfSEP_1 = np.tensordot(a_ccar[:,1], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_1 = np.tensordot(a_ccar[:,1], noise[:8,:,:], axes=(0, 0))
   ccar_filt_hfSEP_2 = np.tensordot(a_ccar[:,2], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_2 = np.tensordot(a_ccar[:,2], noise[:8,:,:], axes=(0, 0))

   # Compute CSP_hil
   hil_csp_title = title_of_run + '_CSP_hil'
   hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
   hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
   hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
   hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
   hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
   hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)
   hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
   hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

   # Compute CCAr_hil
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

   # return the datasets in epoch, channel, time_in_channel - fashion
   return [
      [np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
      [np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
      [np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
      [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
      [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
   ]


### Loop over the different data-modalities per subject and train the respective model
confusion_matrices = []

for i in range(10):
# for i in range(1):

	idx = i

	for hfsep_dat, noise_dat, title in base_modalities_per_subject:

		workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)
		
		for data, labels, run_title in workload:
			print('run_title: %s' % run_title)
			print(data.shape)

			### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
			shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
			print(shuffled_data.shape)
			X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
			print(X_train.shape)

			elm = meet.elm.ClassELM()

			### Train ELM on the extracted features using cv for hyperparameter and then classify test-samples
			elm.cv(X_train, y_train, folds=5)
			if elm.istrained:
				print('Succesfully trained the ELM. Now work on gaining insights and getting performance metrics.')
				print('Performing classification on the test-data now, after optimizing the randomly initialized ELM. Result:')
				confusion_matrix_after_random_initialization = meet.elm.get_conf_matrix(y_test, elm.classify(X_test))
				print(confusion_matrix_after_random_initialization)
				confusion_matrices.append((run_title, confusion_matrix_after_random_initialization))
				with open('/media/christoph/Volume/paper/new_trainings/elm/models/elm_%s' % run_title.replace('/', '-'), 'wb') as target_file: 
				    pickle.dump(elm, target_file)


from numpy import save
print(confusion_matrices)

save('/media/christoph/Volume/paper/new_trainings/elm/confusion_matrices_run_csp', confusion_matrices)