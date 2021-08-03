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
	print('amount of labels: %d' % noise_CSP_0.shape[-1])

	# return the datasets in epoch, channel, time_in_channel - fashion
	return [
		[np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
		[np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((np.ones(hfSEP_CSP_0.shape[-1], dtype=np.int8), np.zeros(noise_CSP_0.shape[-1], dtype=np.int8)), axis=0), csp_title],
		[np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
		[np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((np.ones(hfSEP_CSP_0.shape[-1], dtype=np.int8), np.zeros(noise_CSP_0.shape[-1], dtype=np.int8)), axis=0), hil_csp_title],
		[np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
	]


def create_alex_net(input_len):
	model = model = keras.models.Sequential([ 
		keras.layers.Conv1D(filters=96, kernel_size=(11), strides=(4), activation='relu', input_shape=(input_len,1)), 
		keras.layers.BatchNormalization(), 
		keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
		keras.layers.Conv1D(filters=256, kernel_size=(5), strides=(1), activation='relu', padding="same"), 
		keras.layers.BatchNormalization(), 
		keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
		keras.layers.Conv1D(filters=384, kernel_size=(3), strides=(1), activation='relu', padding="same"), 
		keras.layers.BatchNormalization(), 
		keras.layers.Conv1D(filters=384, kernel_size=(1), strides=(1), activation='relu', padding="same"), 
		keras.layers.BatchNormalization(), 
		keras.layers.Conv1D(filters=256, kernel_size=(1), strides=(1), activation='relu', padding="same"), 
		keras.layers.BatchNormalization(), 
		keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
		keras.layers.Flatten(), 
		keras.layers.Dense(4096, activation='relu'), 
		keras.layers.Dropout(0.5), 
		keras.layers.Dense(4096, activation='relu'), 
		keras.layers.Dropout(0.5), 
		keras.layers.Dense(1024, activation='relu'), 
		keras.layers.Dropout(0.5), 
		keras.layers.Dense(1, activation='sigmoid') 
	])

	METRICS = [
		tf.keras.metrics.TruePositives(),
		tf.keras.metrics.FalsePositives(),
		tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.FalseNegatives(), 
		tf.keras.metrics.BinaryAccuracy(),
		tf.keras.metrics.Precision(),
		tf.keras.metrics.Recall(),
		tf.keras.metrics.AUC(),
		tf.keras.metrics.MeanAbsoluteError(),
	]

	model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
		loss=tf.keras.losses.BinaryCrossentropy(), 
		metrics=METRICS)

	return model

### Loop over the different data-modalities per subject and train the respective model
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
			X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

			X_train = np.expand_dims(X_train, axis=-1)
			X_test = np.expand_dims(X_test, axis=-1)
			X_eval = np.expand_dims(X_eval, axis=-1)

			y_train = y_train.reshape((-1,1))
			y_test = y_test.reshape((-1,1))
			y_eval = y_eval.reshape((-1,1))

			model = create_alex_net(X_train.shape[1])

			history = model.fit(x=X_train, y=y_train, epochs=max_epochs, batch_size=32, 
				validation_data=[X_eval, y_eval], callbacks=[
				tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
					patience=10, restore_best_weights=True)
				]
			)

			model.save('/media/christoph/Volume/paper/new_trainings/alexnet/models/%d_alex_net_on_%s' % (idx, run_title.replace('/', '-')))