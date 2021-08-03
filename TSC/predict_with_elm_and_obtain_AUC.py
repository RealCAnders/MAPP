import sys  
from os import listdir

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 
 
import tensorflow_docs as tfdocs 
import tensorflow_docs.modeling 
import tensorflow_docs.plots 
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import scipy  
from scipy import signal    
from scipy.fft import fftshift    
from scipy.ndimage import convolve1d, convolve    

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

import meet    
from meet.spatfilt import CSP

import numpy as np    
from numpy import save
from numpy import load

import pickle

import matplotlib.pyplot as plt    
from matplotlib.font_manager import FontProperties 

## ### ### ### ###  
## Definition: utility-function / global vars  
## ### ### ### ###  
offset = 1000  
s_rate = 10000  
stim_per_sec = 4  
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]  
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]                                                                                      
                                      
## ### ### ### ###  
## Data loading  
## ### ### ### ###  
hfSEP_win = [50, 450]  
noise_win = [-500, -100]  
intrplt_win = [-80, 30] 
rand_stat = 42

def calculate_hil_features(transformed): 
   # For the transformed data, calclulate respective Hilbert Transform and return
   # real, imaginary, absolute, and angle thereof
   hil_dat = scipy.signal.hilbert(transformed, axis=0) 
   real_hil_dat = np.real(hil_dat) 
   imag_hil_dat = np.imag(hil_dat) 
   abs_hil_dat = np.abs(hil_dat) 
   angle_hil_dat = np.angle(hil_dat) 
   return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0) 

def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
# load data in format of (channel x epoch_length x number of epochs)
   title_of_run = title % (identifier)
   hfSEP = load(hfsep_path % (identifier))
   noise = load(noise_path % (identifier))

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

modalities_fpr = []
modalities_tpr = []
modalities_thresholds = []
modalities_aucs =  []

for subject_id in range(1, 11, 1):
   for modality_pair in [['raw', 0], ['CSP', 1], ['CCAr', 2], ['CSP_hil', 3], ['CCAR_hil', 4]]:
      true_labels = np.asarray([])
      predicted_labels = np.asarray([])
      sample_counter = 0
      elm_model_k00x = load('/media/christoph/Volume1/Masterthesis/elm_models_trained/classes_balanced/elm_k00%d-epoched_intrplt_filt_500_900_kx_%s' % (subject_id, modality_pair[0]), allow_pickle=True)
      data_for_prediction = ['/media/christoph/Volume1/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume1/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
      workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(data_for_prediction[0], data_for_prediction[1], data_for_prediction[2], subject_id)
      data, labels, run_title = workload[modality_pair[1]]
      print(run_title)
      print("Subject_id == %d" % subject_id)
      ### Shuffle and split data // .T is required to switch back to shape of (trial x feature) 
      shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat) 
      print(shuffled_data.shape) 
      X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat) 
      print(X_train.shape)
      print("Testing_data increase from %d to %d" % (sample_counter, sample_counter + X_test.shape[0]))
      sample_counter = sample_counter + X_test.shape[0]
      y_pred_elmk00x = elm_model_k00x._run(X_test)
      true_labels = np.concatenate((true_labels, y_test), axis=0)
      predicted_labels = np.concatenate((predicted_labels, y_pred_elmk00x), axis=0)
      fpr_elm_k00x, tpr_elm_k00x, thresholds_elm_k00x = roc_curve(y_test, y_pred_elmk00x)
      auc_elmk00x = auc(fpr_elm_k00x, tpr_elm_k00x)
      print("AUC for this run: %f" % auc_elmk00x)
      fpr_elm_overall_ccar_hil, tpr_elc_overall_ccar_hil, thresholds_elm_overall_ccar_hil = roc_curve(true_labels, predicted_labels)
      auc_elm_overall_ccar_hil = auc(fpr_elm_overall_ccar_hil, tpr_elc_overall_ccar_hil)
      modalities_fpr.append(fpr_elm_overall_ccar_hil)
      modalities_tpr.append(tpr_elc_overall_ccar_hil)
      modalities_thresholds.append(thresholds_elm_overall_ccar_hil)
      modalities_aucs.append(auc_elm_overall_ccar_hil)

   plt.figure(1) 
   plt.plot([0, 1], [0, 1], 'k--') 
   base_accessor = (subject_id - 1) * 5
   for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAR_hil']):
      plt.plot(modalities_fpr[base_accessor + c], modalities_tpr[base_accessor + c], label='%s (area = %.3f)' % (modality, modalities_aucs[base_accessor + c]))
   plt.xlabel('False positive rate') 
   plt.ylabel('True positive rate') 
   plt.title('ELM ROC curve for SO%d' % subject_id) 
   plt.legend(loc='best')
   plt.gcf().set_size_inches(5.5, 3.725)
   plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_13_04/newest_rocs/elm_roc_s%d.png' % subject_id, dpi=300)
   plt.close('all')